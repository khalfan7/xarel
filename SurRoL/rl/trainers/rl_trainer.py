import copy
import os

import gym
import torch

from agents.factory import make_agent
from components.checkpointer import CheckpointHandler
from components.logger import Logger, WandBLogger, logger
from modules.replay_buffer import HerReplayBuffer, get_buffer_sampler
from modules.samplers import Sampler, RLIFSampler
from utils.general_utils import (AverageMeter, Every, Timer, Until,
                                   set_seed_everywhere)
from utils.mpi import (mpi_gather_experience_episode,
                         mpi_gather_experience_rollots, mpi_sum,
                         update_mpi_config)
from utils.rl_utils import RolloutStorage, get_env_params, init_buffer
from .base_trainer import BaseTrainer


class RLTrainer(BaseTrainer):
    def _setup(self):
        self._setup_env()       # Environment
        self._setup_buffer()    # Relay buffer
        self._setup_agent()     # Agent
        self._setup_sampler()   # Sampler
        self._setup_logger()    # Logger
        self._setup_misc()      # MISC

        if self.is_chef:
            self.termlog.info('Setup done')

    def _setup_env(self):
        self.train_env = gym.make(self.cfg.task)
        # self.eval_env = gym.make(self.cfg.task)
        self.env_params = get_env_params(self.train_env, self.cfg)
        
    def _setup_buffer(self):
        self.buffer_sampler = get_buffer_sampler(self.train_env, self.cfg.agent.sampler)
        self.buffer = HerReplayBuffer(buffer_size=self.cfg.replay_buffer_capacity, env_params=self.env_params,
                            batch_size=self.cfg.batch_size, sampler=self.buffer_sampler)
        self.demo_buffer = HerReplayBuffer(buffer_size=self.cfg.replay_buffer_capacity, env_params=self.env_params,
                            batch_size=self.cfg.batch_size, sampler=self.buffer_sampler)
        
    def _setup_agent(self):
        self.agent = make_agent(self.env_params, self.buffer_sampler, self.cfg.agent)

    def _setup_sampler(self):
        self.train_sampler = Sampler(self.train_env, self.agent, self.env_params['max_timesteps'])
        self.eval_sampler = Sampler(self.train_env, self.agent, self.env_params['max_timesteps'])

    def _setup_logger(self):
        update_mpi_config(self.cfg)
        if self.is_chef:
            exp_name = f"{self.cfg.task}_{self.cfg.agent.name}_demo{self.cfg.num_demo}_seed{self.cfg.seed}"
            if self.cfg.postfix is not None:
                exp_name =  exp_name + '_' + str(self.cfg.postfix) 
            if self.cfg.use_wb:
                self.wb = WandBLogger(exp_name=exp_name, project_name=self.cfg.project_name, entity=self.cfg.entity_name, \
                        path=self.work_dir, conf=self.cfg)
            self.logger = Logger(self.work_dir)
            self.termlog = logger
        else: 
            self.wb, self.logger, self.termlog = None, None, None
    
    def _setup_misc(self):
        init_buffer(self.cfg, self.buffer, self.agent, normalize=False) # important for awac, amp
        init_buffer(self.cfg, self.demo_buffer, self.agent, normalize=True)

        if self.is_chef:
            self.model_dir = self.work_dir / 'model'
            self.model_dir.mkdir(exist_ok=True)
            for file in os.listdir(self.model_dir):
                os.remove(self.model_dir / file)

        self.device = torch.device(self.cfg.device)
        self.timer = Timer()
        self._global_step = 0
        self._global_episode = 0
        set_seed_everywhere(self.cfg.seed)
    
    def train(self):
        n_train_episodes = int(self.cfg.n_train_steps / self.env_params['max_timesteps'])
        n_eval_episodes = int(n_train_episodes / self.cfg.n_eval) * self.cfg.mpi.num_workers
        n_save_episodes = int(n_train_episodes / self.cfg.n_save) * self.cfg.mpi.num_workers
        n_log_episodes = int(n_train_episodes / self.cfg.n_log) * self.cfg.mpi.num_workers

        assert n_save_episodes > n_eval_episodes
        if n_save_episodes % n_eval_episodes != 0:
            n_save_episodes = int(n_save_episodes / n_eval_episodes) * n_eval_episodes

        train_until_episode = Until(n_train_episodes)
        save_every_episodes = Every(n_save_episodes)
        eval_every_episodes = Every(n_eval_episodes)
        log_every_episodes = Every(n_log_episodes)
        seed_until_steps = Until(self.cfg.n_seed_steps)

        if self.is_chef:
            self.termlog.info('Starting training')
        while train_until_episode(self.global_episode):
            self._train_episode(log_every_episodes, seed_until_steps)

            if eval_every_episodes(self.global_episode):
                score = self.eval()

            if not self.cfg.dont_save and save_every_episodes(self.global_episode) and self.is_chef:
                filename =  CheckpointHandler.get_ckpt_name(self.global_episode)
                # TODO(tao): expose scoring metric
                CheckpointHandler.save_checkpoint({
                    'episode': self.global_episode,
                    'global_step': self.global_step,
                    'state_dict': self.agent.state_dict(),
                    'o_norm': self.agent.o_norm,
                    'g_norm': self.agent.g_norm,
                    'score': score,
                }, self.model_dir, filename)
                self.termlog.info(f'Save checkpoint to {os.path.join(self.model_dir, filename)}')

    def _train_episode(self, log_every_episodes, seed_until_steps):
        # sync network parameters across workers
        if self.use_multiple_workers > 1:
            self.agent.sync_networks()

        self.timer.reset()
        batch_time = AverageMeter()
        ep_start_step = self.global_step
        metrics = None

        # collect experience
        rollout_storage = RolloutStorage()
        episode, rollouts, env_steps = self.train_sampler.sample_episode(is_train=True, render=False, random_act=seed_until_steps(ep_start_step))
        if self.use_multiple_workers:
            transitions_batch = mpi_gather_experience_episode(rollouts)
        else:
            transitions_batch = rollouts

        # update status
        rollout_storage.append(episode)
        rollout_status = rollout_storage.rollout_stats()
        self._global_step += int(mpi_sum(env_steps))
        self._global_episode += int(mpi_sum(1))

        # save to buffer
        self.buffer.store_episode(copy.deepcopy(transitions_batch))
        self.agent.update_normalizer(copy.deepcopy(transitions_batch))

        # update policy
        if not seed_until_steps(ep_start_step):
            if self.is_chef:
                metrics = self.agent.update(self.buffer, self.demo_buffer)
            if self.use_multiple_workers:
                self.agent.sync_networks()
        
        # Periodic CUDA cache clearing every 50 episodes to prevent "reserved" memory growth
        if self.global_episode % 50 == 0 and torch.cuda.is_available():
            torch.cuda.empty_cache()

        # log results
        if metrics is not None and log_every_episodes(self.global_episode) and self.is_chef:
            elapsed_time, total_time = self.timer.reset()
            batch_time.update(elapsed_time)
            togo_train_time = batch_time.avg * (self.cfg.n_train_steps - ep_start_step) / env_steps

            # Log memory usage periodically with buffer stats
            try:
                import psutil
                if torch.cuda.is_available():
                    gpu_mem_allocated = torch.cuda.memory_allocated(0) / 1024**2  # MB
                    gpu_mem_reserved = torch.cuda.memory_reserved(0) / 1024**2  # MB
                    ram_used = psutil.Process().memory_info().rss / 1024**2  # MB
                    ram_percent = psutil.virtual_memory().percent
                    
                    # Calculate replay buffer fill fraction
                    buffer_size = self.buffer.current_size if hasattr(self.buffer, 'current_size') else 0
                    buffer_capacity = self.cfg.replay_buffer_capacity
                    buffer_fraction = buffer_size / buffer_capacity if buffer_capacity > 0 else 0
                    
                    self.termlog.info(f'[Step {self.global_step}] GPU: {gpu_mem_allocated:.0f}MB alloc, {gpu_mem_reserved:.0f}MB reserved | RAM: {ram_used:.0f}MB ({ram_percent:.1f}% sys) | Buffer: {buffer_fraction:.2%} full ({buffer_size}/{buffer_capacity})')
            except ImportError:
                pass
            
            self.logger.log_metrics(metrics, self.global_step, ty='train')
            with self.logger.log_and_dump_ctx(self.global_step, ty='train') as log:
                log('fps', env_steps / elapsed_time)
                log('total_time', total_time)
                log('episode_reward', rollout_status.avg_reward)
                log('episode_length', env_steps)
                log('episode_sr', rollout_status.avg_success_rate)
                log('episode', self.global_episode)
                log('step', self.global_step)
                log('ETA', togo_train_time)
            if self.cfg.use_wb:
                self.wb.log_outputs(metrics, None, log_images=False, step=self.global_step, is_train=True)

    def eval(self):
        '''Eval agent.'''
        # Log memory BEFORE eval to detect eval-triggered spikes
        self._eval_mem_pre = {}
        try:
            import psutil
            if torch.cuda.is_available():
                self._eval_mem_pre['gpu_alloc'] = torch.cuda.memory_allocated(0) / 1024**2
                self._eval_mem_pre['gpu_reserved'] = torch.cuda.memory_reserved(0) / 1024**2
                self._eval_mem_pre['ram'] = psutil.Process().memory_info().rss / 1024**2
                self._eval_mem_pre['ram_pct'] = psutil.virtual_memory().percent
                self.termlog.info(f'[EVAL START] Step {self.global_step}: RAM={self._eval_mem_pre["ram"]:.0f}MB ({self._eval_mem_pre["ram_pct"]:.1f}%), GPU alloc={self._eval_mem_pre["gpu_alloc"]:.0f}MB, reserved={self._eval_mem_pre["gpu_reserved"]:.0f}MB')
        except ImportError:
            pass
        
        eval_rollout_storage = RolloutStorage()
        for _ in range(self.cfg.n_eval_episodes):
            episode, _, env_steps = self.eval_sampler.sample_episode(is_train=False, render=False)
            eval_rollout_storage.append(episode)
        rollout_status = eval_rollout_storage.rollout_stats()
        if self.use_multiple_workers:
            rollout_status = mpi_gather_experience_rollots(rollout_status)
            for key, value in rollout_status.items():
                rollout_status[key] = value.mean()

        if self.is_chef:
            if self.cfg.use_wb:
                self.wb.log_outputs(rollout_status, eval_rollout_storage, log_images=True, step=self.global_step)
            with self.logger.log_and_dump_ctx(self.global_step, ty='eval') as log:
                log('episode_sr', rollout_status.avg_success_rate)
                log('episode_reward', rollout_status.avg_reward)
                log('episode_length', env_steps)
                log('episode', self.global_episode)
                log('step', self.global_step)

        del eval_rollout_storage
        
        # Aggressive memory cleanup after evaluation
        import gc
        import psutil
        
        # Log memory before cleanup
        if self.is_chef and torch.cuda.is_available():
            gpu_mem_allocated = torch.cuda.memory_allocated(0) / 1024**2  # MB
            gpu_mem_reserved = torch.cuda.memory_reserved(0) / 1024**2  # MB
            ram_used = psutil.Process().memory_info().rss / 1024**2  # MB
            self.termlog.info(f'Memory before cleanup: GPU alloc={gpu_mem_allocated:.1f}MB, reserved={gpu_mem_reserved:.1f}MB, RAM={ram_used:.1f}MB')
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Log memory after cleanup
        if self.is_chef and torch.cuda.is_available():
            gpu_mem_allocated = torch.cuda.memory_allocated(0) / 1024**2  # MB
            gpu_mem_reserved = torch.cuda.memory_reserved(0) / 1024**2  # MB
            ram_used = psutil.Process().memory_info().rss / 1024**2  # MB
            self.termlog.info(f'Memory after cleanup: GPU alloc={gpu_mem_allocated:.1f}MB, reserved={gpu_mem_reserved:.1f}MB, RAM={ram_used:.1f}MB')
        
        return rollout_status.avg_success_rate

    def eval_ckpt(self):
        '''Eval checkpoint.'''
        if self.is_chef:
            CheckpointHandler.load_checkpoint(
                self.cfg.ckpt_dir, self.agent, self.device, self.cfg.ckpt_episode
            )
            avg_success_rate = self.eval()
            self.termlog.info(f'Successful rate: {avg_success_rate}')
                
    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def is_chef(self):
        return self.cfg.mpi.is_chef

    @property
    def use_multiple_workers(self):
        return self.cfg.mpi.num_workers > 1

class RLIFRLTrainer(RLTrainer):
    def _setup_sampler(self):
        self.train_sampler = RLIFSampler(self.train_env, self.agent, self.env_params['max_timesteps'])
        self.eval_sampler = Sampler(self.train_env, self.agent, self.env_params['max_timesteps'])

    def train(self):
        n_train_episodes = int(self.cfg.n_train_steps / self.env_params['max_timesteps'])
        n_eval_episodes = int(n_train_episodes / self.cfg.n_eval) * self.cfg.mpi.num_workers
        n_save_episodes = int(n_train_episodes / self.cfg.n_save) * self.cfg.mpi.num_workers
        n_log_episodes = int(n_train_episodes / self.cfg.n_log) * self.cfg.mpi.num_workers

        assert n_save_episodes > n_eval_episodes
        if n_save_episodes % n_eval_episodes != 0:
            n_save_episodes = int(n_save_episodes / n_eval_episodes) * n_eval_episodes

        train_until_episode = Until(n_train_episodes)
        save_every_episodes = Every(n_save_episodes)
        eval_every_episodes = Every(n_eval_episodes)
        log_every_episodes = Every(n_log_episodes)
        seed_until_steps = Until(self.cfg.n_seed_steps)

        if self.is_chef:
            self.termlog.info('Starting training')
        while train_until_episode(self.global_episode):

            self._train_episode(log_every_episodes, seed_until_steps)

            if eval_every_episodes(self.global_episode):
                score = self.eval()

            if not self.cfg.dont_save and save_every_episodes(self.global_episode) and self.is_chef:
                filename =  CheckpointHandler.get_ckpt_name(self.global_episode)
                # TODO(tao): expose scoring metric
                CheckpointHandler.save_checkpoint({
                    'episode': self.global_episode,
                    'global_step': self.global_step,
                    'state_dict': self.agent.state_dict(),
                    'o_norm': self.agent.o_norm,
                    'g_norm': self.agent.g_norm,
                    'score': score,
                }, self.model_dir, filename)
                self.termlog.info(f'Save checkpoint to {os.path.join(self.model_dir, filename)}')

    def eval(self):
        '''Eval agent.'''
        # Log memory BEFORE eval to detect eval-triggered spikes
        self._eval_mem_pre = {}
        try:
            import psutil
            if torch.cuda.is_available():
                self._eval_mem_pre['gpu_alloc'] = torch.cuda.memory_allocated(0) / 1024**2
                self._eval_mem_pre['gpu_reserved'] = torch.cuda.memory_reserved(0) / 1024**2
                self._eval_mem_pre['ram'] = psutil.Process().memory_info().rss / 1024**2
                self._eval_mem_pre['ram_pct'] = psutil.virtual_memory().percent
                self.termlog.info(f'[EVAL START] Step {self.global_step}: RAM={self._eval_mem_pre["ram"]:.0f}MB ({self._eval_mem_pre["ram_pct"]:.1f}%), GPU alloc={self._eval_mem_pre["gpu_alloc"]:.0f}MB, reserved={self._eval_mem_pre["gpu_reserved"]:.0f}MB')
        except ImportError:
            pass
        
        eval_rollout_storage = RolloutStorage()
        for _ in range(self.cfg.n_eval_episodes):
            episode, _, env_steps = self.eval_sampler.sample_episode(is_train=False, render=False)
            eval_rollout_storage.append(episode)
        rollout_status = eval_rollout_storage.rollout_stats()
        if self.use_multiple_workers:
            rollout_status = mpi_gather_experience_rollots(rollout_status)
            for key, value in rollout_status.items():
                rollout_status[key] = value.mean()

        if self.is_chef:
            if self.cfg.use_wb:
                self.wb.log_outputs(rollout_status, eval_rollout_storage, log_images=True, step=self.global_step)
            with self.logger.log_and_dump_ctx(self.global_step, ty='eval') as log:
                log('episode_sr', rollout_status.avg_success_rate)
                log('episode_reward', rollout_status.avg_reward)
                log('episode_length', env_steps)
                log('episode', self.global_episode)
                log('step', self.global_step)

        del eval_rollout_storage
        
        # Aggressive memory cleanup after evaluation
        import gc
        try:
            import psutil
            # Log memory before cleanup
            if self.is_chef and torch.cuda.is_available():
                gpu_alloc_post = torch.cuda.memory_allocated(0) / 1024**2  # MB
                gpu_reserved_post = torch.cuda.memory_reserved(0) / 1024**2  # MB
                ram_post = psutil.Process().memory_info().rss / 1024**2  # MB
                ram_pct_post = psutil.virtual_memory().percent
                
                # Calculate deltas from pre-eval
                if hasattr(self, '_eval_mem_pre') and self._eval_mem_pre:
                    ram_delta = ram_post - self._eval_mem_pre['ram']
                    gpu_alloc_delta = gpu_alloc_post - self._eval_mem_pre['gpu_alloc']
                    gpu_reserved_delta = gpu_reserved_post - self._eval_mem_pre['gpu_reserved']
                    self.termlog.info(f'[EVAL END] Before cleanup: RAM={ram_post:.0f}MB (+{ram_delta:.0f}, {ram_pct_post:.1f}%), GPU alloc={gpu_alloc_post:.0f}MB (+{gpu_alloc_delta:.0f}), reserved={gpu_reserved_post:.0f}MB (+{gpu_reserved_delta:.0f})')
                else:
                    self.termlog.info(f'[EVAL END] Before cleanup: RAM={ram_post:.0f}MB ({ram_pct_post:.1f}%), GPU alloc={gpu_alloc_post:.0f}MB, reserved={gpu_reserved_post:.0f}MB')
        except ImportError:
            pass
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Log memory after cleanup
        try:
            if self.is_chef and torch.cuda.is_available():
                gpu_alloc_clean = torch.cuda.memory_allocated(0) / 1024**2  # MB
                gpu_reserved_clean = torch.cuda.memory_reserved(0) / 1024**2  # MB
                ram_clean = psutil.Process().memory_info().rss / 1024**2  # MB
                ram_pct_clean = psutil.virtual_memory().percent
                
                cleanup_saved_ram = ram_post - ram_clean if 'ram_post' in locals() else 0
                cleanup_saved_gpu = gpu_reserved_post - gpu_reserved_clean if 'gpu_reserved_post' in locals() else 0
                self.termlog.info(f'[EVAL END] After cleanup: RAM={ram_clean:.0f}MB (freed {cleanup_saved_ram:.0f}MB, {ram_pct_clean:.1f}%), GPU reserved={gpu_reserved_clean:.0f}MB (freed {cleanup_saved_gpu:.0f}MB)')
        except (ImportError, NameError):
            pass
        
        return rollout_status.avg_success_rate
