import os
import sys
import numpy as np
import gymnasium
import hydra
from pathlib import Path
from omegaconf import DictConfig

# Add SurRoL to path
sys.path.insert(0, str(Path(__file__).parent.parent / "SurRoL"))

from stable_baselines3 import SAC
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env

# Register SurRoL environments
import surrol.gym


def make_env(task_name, rank=0, seed=0):
    def _init():
        env = gymnasium.make(task_name, render_mode=None)
        return env
    return _init

@hydra.main(version_base=None, config_path="./configs", config_name="train_sb3")
def main(cfg: DictConfig):
    script_dir = Path(__file__).parent
    output_dir = script_dir / "outputs" / cfg.task / f"seed_{cfg.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create vectorized environment
    env = make_vec_env(
        make_env(cfg.task, seed=cfg.seed),
        n_envs=cfg.n_envs,
        vec_env_cls=SubprocVecEnv,
        seed=cfg.seed
    )
    
    # Normalize observations
    if cfg.normalize:
        env = VecNormalize(
            env,
            norm_obs=cfg.get('norm_obs', True),
            norm_reward=cfg.get('norm_reward', False),
            clip_obs=cfg.get('clip_obs', 10.0),  # SB3 default is 10.0
            clip_reward=cfg.get('clip_reward', 10.0),
        )
    
    # Create eval environment
    eval_env = make_vec_env(
        make_env(cfg.task, seed=cfg.seed + 1000),
        n_envs=1,
        seed=cfg.seed + 1000
    )
    if cfg.normalize:
        eval_env = VecNormalize(
            eval_env,
            training=False,  # Don't update stats during eval
            norm_obs=cfg.get('norm_obs', True),
            norm_reward=False,
            clip_obs=cfg.get('clip_obs', 10.0),
        )
    
    # HER-specific configuration
    her_kwargs = dict(
        n_sampled_goal=cfg.her.n_sampled_goal,
        goal_selection_strategy=cfg.her.strategy,
        handle_timeout_termination=cfg.her.handle_timeout_termination,
    )

    policy_kwargs = dict(
        net_arch=list(cfg.policy_kwargs.net_arch)
    )
    
    # SAC + HER
    model = SAC(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=cfg.learning_rate,
        buffer_size=cfg.buffer_size,
        learning_starts=cfg.learning_starts,
        batch_size=cfg.batch_size,
        tau=cfg.tau,
        gamma=cfg.gamma,
        train_freq=cfg.train_freq,
        gradient_steps=cfg.gradient_steps,
        ent_coef=cfg.ent_coef,
        target_update_interval=cfg.target_update_interval,
        target_entropy=cfg.target_entropy,
        use_sde=cfg.use_sde,
        policy_kwargs=policy_kwargs,
        replay_buffer_class=HerReplayBuffer,
        replay_buffer_kwargs=her_kwargs,
        verbose=1,
        tensorboard_log=str(output_dir / "tensorboard"),
        device=cfg.device,
        seed=cfg.seed,
    )
    
    # Callbacks
    callbacks = []
    
    # Checkpoint callback - save model periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=cfg.save_freq,
        save_path=str(output_dir / "checkpoints"),
        name_prefix="sac_her",
        save_replay_buffer=False, 
        save_vecnormalize=cfg.normalize,
    )
    callbacks.append(checkpoint_callback)
    
    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output_dir / "best_model"),
        log_path=str(output_dir / "eval_logs"),
        eval_freq=cfg.eval_freq,
        n_eval_episodes=cfg.n_eval_episodes,
        deterministic=True,
        render=False,
    )
    callbacks.append(eval_callback)
    
    # Train the model
    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback=callbacks,
        log_interval=cfg.log_interval,
        tb_log_name=f"{cfg.task}_seed{cfg.seed}",
        reset_num_timesteps=True,
        progress_bar=True,
    )
    
    # Save final model
    final_model_path = output_dir / "final_model"
    model.save(str(final_model_path))
    
    # Save normalization stats before closing
    if cfg.normalize:
        vec_normalize_path = output_dir / "vec_normalize.pkl"
        env.save(str(vec_normalize_path))
        eval_env.save(str(output_dir / "vec_normalize_eval.pkl"))
    
    # Close environments
    env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
