"""
Improved SAC+HER training script with SB3 best practices and RL tips & tricks.

Key improvements:
1. Better HER configuration (GoalSelectionStrategy enum support)
2. Proper VecNormalize synchronization between train/eval
3. Enhanced logging and monitoring with Monitor wrapper
4. Gradient steps optimization for sample efficiency
5. Better checkpoint management (adjusted for parallel envs)
6. Learning rate scheduling option
7. Proper warmup handling with use_sde_at_warmup
8. Policy-only saving for efficient inference
9. Environment validation with check_env()

Environment Requirements (Custom Environments):
- Must follow Gymnasium interface (gym.Env)
- For goal-based RL: Dict observation space with keys:
  * 'observation': current state
  * 'achieved_goal': what was achieved
  * 'desired_goal': what we want to achieve
- Must implement compute_reward(achieved_goal, desired_goal, info) method
- SurRoL environments already satisfy these requirements

Best Practices for Custom Environments:
1. ALWAYS normalize observations (use VecNormalize)
2. NORMALIZE action space to [-1, 1] (symmetric, centered at 0)
3. Handle timeout (truncated=True) separately from termination
4. Start with shaped rewards, not purely sparse
5. Avoid breaking Markov assumption (include history if needed)
6. Use check_env() to validate your environment
7. Test with random actions first to verify environment works
"""

import os
import sys
import numpy as np
import gymnasium
import hydra
from pathlib import Path
from omegaconf import DictConfig
import torch

# Add SurRoL to path
sys.path.insert(0, str(Path(__file__).parent.parent / "SurRoL"))

from stable_baselines3 import SAC
from stable_baselines3.her.goal_selection_strategy import GoalSelectionStrategy
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor

# Import HER replay buffer
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer

# Register SurRoL environments
import surrol.gym


class TimeLimitWithComputeReward(gymnasium.Wrapper):
    """
    TimeLimit wrapper that properly forwards compute_reward for HER compatibility.
    
    Gymnasium's default TimeLimit wrapper doesn't expose compute_reward,
    which breaks SB3's check_env() for goal-based environments.
    """
    def __init__(self, env, max_episode_steps=None):
        super().__init__(env)
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = 0
    
    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self._elapsed_steps += 1
        
        if self._elapsed_steps >= self._max_episode_steps:
            truncated = True
        
        return observation, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)
    
    def compute_reward(self, achieved_goal, desired_goal, info):
        """Forward compute_reward to the underlying environment, recursively unwrapping if needed."""
        env = self.env
        # Recursively unwrap to find the environment with compute_reward
        while hasattr(env, 'env') and not hasattr(env, 'compute_reward'):
            env = env.env
        return env.compute_reward(achieved_goal, desired_goal, info)


def make_env(task_name, rank=0, seed=0):
    """
    Create a single environment instance.
    
    Requirements for custom environments (SB3):
    - Inherit from gym.Env
    - Implement: reset(), step(), render(), close()
    - Define: action_space, observation_space (gym.spaces objects)
    - step() returns: observation, reward, terminated, truncated, info
    - reset() returns: observation, info
    
    For goal-based RL with HER:
    - observation_space must be gym.spaces.Dict with keys:
      'observation', 'achieved_goal', 'desired_goal'
    - Must implement: compute_reward(achieved_goal, desired_goal, info)
    
    Best Practices for Custom Environments:
    1. NORMALIZE observation space (critical for RL performance!)
       - Use VecNormalize wrapper (applied later in this script)
       - Know your observation boundaries
    
    2. NORMALIZE action space and make it SYMMETRIC [-1, 1]
       - Most RL algorithms use Gaussian distribution centered at 0
       - Unnormalized actions can silently break learning
       - Rescale actions inside your environment if needed
    
    3. Handle TIMEOUT separately from TERMINATION
       - Return truncated=True for timeout (max steps reached)
       - Return terminated=True for actual task completion/failure
       - Critical for proper value estimation in RL
       - TimeLimit wrapper handles this automatically
    
    4. Use SHAPED REWARDS initially (informative, not sparse)
       - Start with dense rewards during development
       - Can transition to sparse once working
    
    5. Avoid breaking MARKOV assumption
       - If there's time delay, provide history of observations
       - Each observation should contain enough info for decision
    
    Args:
        task_name: Registered environment ID (e.g., 'GauzeRetrieveRL-v0')
        rank: Environment rank (for parallel envs)
        seed: Random seed
    
    Returns:
        Function that creates the environment
    """
    def _init():
        # Create env with automatic TimeLimit wrapper first
        env = gymnasium.make(task_name, render_mode=None)
        
        # If it has a TimeLimit wrapper, unwrap it and apply our custom one
        from gymnasium.wrappers import TimeLimit
        if isinstance(env, TimeLimit):
            max_steps = env._max_episode_steps
            # Unwrap the TimeLimit to get the base environment
            env = env.env
            # Apply our custom TimeLimit wrapper that forwards compute_reward
            env = TimeLimitWithComputeReward(env, max_episode_steps=max_steps)
        
        env = Monitor(env)  # Provides episode statistics
        return env
    return _init


def linear_schedule(initial_value: float):
    """
    Linear learning rate schedule (decay from initial_value to 0).
    
    Useful for fine-tuning later in training.
    
    Args:
        initial_value: Initial learning rate
    
    Returns:
        Schedule function that takes progress_remaining (1.0 to 0.0)
    """
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

@hydra.main(version_base=None, config_path="./configs", config_name="train_sb3")
def main(cfg: DictConfig):
    """Main training function with SB3 best practices."""
    
    # Setup paths
    script_dir = Path(__file__).parent
    output_dir = script_dir / "outputs" / cfg.task / f"seed_{cfg.seed}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Print training configuration
    print(f"{'='*70}")
    print(f"SAC + HER Training Configuration")
    print(f"{'='*70}")
    print(f"Task:              {cfg.task}")
    print(f"Total timesteps:   {cfg.total_timesteps:,}")
    print(f"Parallel envs:     {cfg.n_envs}")
    print(f"Buffer size:       {cfg.buffer_size:,}")
    print(f"Batch size:        {cfg.batch_size}")
    print(f"Learning rate:     {cfg.learning_rate}")
    print(f"Gamma:             {cfg.gamma}")
    print(f"Tau:               {cfg.tau}")
    print(f"Gradient steps:    {cfg.gradient_steps}")
    print(f"HER strategy:      {cfg.her.strategy}")
    print(f"HER n_goals:       {cfg.her.n_sampled_goal}")
    print(f"Normalize obs:     {cfg.normalize}")
    print(f"Device:            {cfg.device}")
    print(f"Seed:              {cfg.seed}")
    print(f"Memory optimize:   {cfg.get('optimize_memory_usage', True)}")
    print(f"Output dir:        {output_dir}")
    print(f"{'='*70}\n")
    
    # Environment validation 
    if cfg.get('check_env', False):
        from stable_baselines3.common.env_checker import check_env
        from gymnasium.wrappers import TimeLimit
        print("Validating environment with SB3 env_checker...")
        test_env = gymnasium.make(cfg.task, render_mode=None)
        
        # Replace TimeLimit wrapper with our custom one that forwards compute_reward
        if isinstance(test_env, TimeLimit):
            max_steps = test_env._max_episode_steps
            test_env = test_env.env  # Unwrap
            test_env = TimeLimitWithComputeReward(test_env, max_episode_steps=max_steps)
        
        check_env(test_env, warn=True)
        test_env.close()
        print("✓ Environment validation passed\n")
    
    # Create vectorized environment
    print(f"Creating {cfg.n_envs} parallel environments...")
    env = make_vec_env(
        make_env(cfg.task, seed=cfg.seed),
        n_envs=cfg.n_envs,
        vec_env_cls=SubprocVecEnv if cfg.n_envs > 1 else DummyVecEnv,
        seed=cfg.seed
    )
    
    # Normalize observations
    if cfg.normalize:
        env = VecNormalize(
            env,
            norm_obs=cfg.get('norm_obs', True),
            norm_reward=cfg.get('norm_reward', False),  # NEVER normalize sparse rewards
            clip_obs=cfg.get('clip_obs', 200.0),  # More generous clipping
            clip_reward=cfg.get('clip_reward', 200.0),
            gamma=cfg.gamma,  # Match discount factor
        )
        print(f"✓ VecNormalize enabled: norm_obs={cfg.get('norm_obs', True)}, "
              f"norm_reward={cfg.get('norm_reward', False)}")
    
    # Create separate evaluation environment
    print("Creating evaluation environment...")
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
            norm_reward=False,  # Never normalize eval rewards
            clip_obs=cfg.get('clip_obs', 200.0),
        )
    
    # HER-specific configuration
    # Map strategy string to enum for compatibility
    goal_strategy = cfg.her.strategy
    if isinstance(goal_strategy, str):
        strategy_map = {
            'future': GoalSelectionStrategy.FUTURE,
            'final': GoalSelectionStrategy.FINAL,
            'episode': GoalSelectionStrategy.EPISODE,
        }
        goal_strategy = strategy_map.get(goal_strategy.lower(), goal_strategy)
    
    her_kwargs = dict(
        n_sampled_goal=cfg.her.n_sampled_goal,
        goal_selection_strategy=goal_strategy,
        handle_timeout_termination=cfg.her.handle_timeout_termination,
        copy_info_dict=cfg.her.get('copy_info_dict', False),  # Only if compute_reward needs info
    )
    
    replay_buffer_class = HerReplayBuffer

    # Policy network configuration
    policy_kwargs = dict(
        net_arch=list(cfg.policy_kwargs.net_arch),
        activation_fn=torch.nn.ReLU,  # SAC default, good for continuous control
        n_critics=cfg.policy_kwargs.get('n_critics', 2),  # Double Q-learning
        share_features_extractor=cfg.policy_kwargs.get('share_features_extractor', False),
    )
    
    # Learning rate schedule 
    if cfg.get('use_lr_schedule', False):
        learning_rate = linear_schedule(cfg.learning_rate)
        print(f"✓ Using linear LR schedule from {cfg.learning_rate} to 0")
    else:
        learning_rate = cfg.learning_rate
    
    # Initialize SAC + HER 
    print("Initializing SAC + HER model...")
    model = SAC(
        policy="MultiInputPolicy",  # Required for Dict observation space
        env=env,
        learning_rate=learning_rate,
        buffer_size=cfg.buffer_size,
        learning_starts=cfg.learning_starts,
        batch_size=cfg.batch_size,
        tau=cfg.tau,
        gamma=cfg.gamma,
        train_freq=cfg.train_freq,
        gradient_steps=cfg.gradient_steps,  # -1 = match rollout steps (more efficient)
        ent_coef=cfg.ent_coef,  # 'auto' to learn entropy coefficient
        target_update_interval=cfg.target_update_interval,
        target_entropy=cfg.target_entropy,  # 'auto' = -dim(action_space)
        use_sde=cfg.use_sde,
        sde_sample_freq=cfg.get('sde_sample_freq', -1),
        use_sde_at_warmup=cfg.get('use_sde_at_warmup', False),
        policy_kwargs=policy_kwargs,
        replay_buffer_class=replay_buffer_class,
        replay_buffer_kwargs=her_kwargs,
        optimize_memory_usage=cfg.get('optimize_memory_usage', True),  # More efficient memory usage
        verbose=1,
        tensorboard_log=str(output_dir / "tensorboard"),
        device=cfg.device,
        seed=cfg.seed,
    )
    print(f"✓ Model initialized with {sum(p.numel() for p in model.policy.parameters()):,} parameters\n")
    
    # Setup callbacks
    print("Configuring callbacks...")
    callbacks = []
    
    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=max(cfg.save_freq // cfg.n_envs, 1),  # Adjust for parallel envs
        save_path=str(output_dir / "checkpoints"),
        name_prefix="sac_her",
        save_replay_buffer=cfg.get('save_replay_buffer', False),  # Can be large
        save_vecnormalize=cfg.normalize,
        verbose=1,
    )
    callbacks.append(checkpoint_callback)
    print(f"  ✓ Checkpoint every {cfg.save_freq:,} steps")
    
    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(output_dir / "best_model"),
        log_path=str(output_dir / "eval_logs"),
        eval_freq=max(cfg.eval_freq // cfg.n_envs, 1),  # Adjust for parallel envs
        n_eval_episodes=cfg.n_eval_episodes,
        deterministic=True,  # Use deterministic policy for fair evaluation
        render=False,
        verbose=1,
        warn=True,
    )
    callbacks.append(eval_callback)
    print(f"  ✓ Evaluation every {cfg.eval_freq:,} steps ({cfg.n_eval_episodes} episodes)")
    
    # VecNormalize stats are automatically synced by model.learn()
    
    # Train the model
    print(f"\n{'='*70}")
    print(f"Starting training for {cfg.total_timesteps:,} timesteps")
    print(f"Random exploration for first {cfg.learning_starts:,} steps")
    print(f"{'='*70}\n")
    
    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback=callbacks,
        log_interval=cfg.log_interval,  # Log every N episodes
        tb_log_name=f"{cfg.task}_seed{cfg.seed}",
        reset_num_timesteps=True,
        progress_bar=True,
    )
    
    print(f"\n{'='*70}")
    print(f"Training completed!")
    print(f"{'='*70}\n")
    
    # Save final model
    print("Saving final model and artifacts...")
    final_model_path = output_dir / "final_model"
    model.save(str(final_model_path))
    print(f"  ✓ Final model saved: {final_model_path}")
    
    # Save policy separately for inference without env 
    if cfg.get('save_policy_only', True):
        policy_path = output_dir / "final_policy"
        model.policy.save(str(policy_path))
        print(f"  ✓ Policy-only saved: {policy_path} (for inference)")
    
    # Save replay buffer - check
    if cfg.get('save_final_replay_buffer', False):
        replay_buffer_path = output_dir / "final_replay_buffer.pkl"
        model.save_replay_buffer(str(replay_buffer_path))
        print(f"  ✓ Replay buffer saved: {replay_buffer_path}")
    
    # Save normalization stats before closing
    if cfg.normalize:
        vec_normalize_path = output_dir / "vec_normalize.pkl"
        env.save(str(vec_normalize_path))
        print(f"  ✓ VecNormalize stats saved: {vec_normalize_path}")
    
    # Close environments
    env.close()
    eval_env.close()
    
    print(f"\n{'='*70}")
    print(f"All outputs saved to: {output_dir}")
    print(f"View training progress: tensorboard --logdir {output_dir / 'tensorboard'}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
