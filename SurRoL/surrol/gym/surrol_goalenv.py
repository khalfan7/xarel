import gymnasium as gym
from gymnasium import error
from surrol.gym.surrol_env import SurRoLEnv


class SurRoLGoalEnv(SurRoLEnv):
    """
    A gymnasium GoalEnv wrapper for SurRoL.
    Migrated from gym to gymnasium for compatibility with Stable-Baselines3 2.0+
    """

    def reset(self, seed=None, options=None):
        # Handle seed parameter (new gymnasium API)
        if seed is not None:
            self.seed(seed)
        
        # Enforce that each GoalEnv uses a Goal-compatible observation space.
        if not isinstance(self.observation_space, gym.spaces.Dict):
            raise error.Error('GoalEnv requires an observation space of type gymnasium.spaces.Dict')
        for key in ['observation', 'achieved_goal', 'desired_goal']:
            if key not in self.observation_space.spaces:
                raise error.Error('GoalEnv requires the "{}" key to be part of the observation dictionary.'.format(key))
        
        # Call parent reset (returns obs only in old API)
        obs = super().reset()
        info = {}
        return obs, info
