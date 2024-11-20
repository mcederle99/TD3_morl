import gymnasium as gym
from gymnasium import Wrapper


class TwoDimRewardWrapper(Wrapper):
    """
    A wrapper for the HalfCheetah-v5 environment to return a 2-dimensional reward.

    The reward vector contains:
    1. The forward velocity of the cheetah.
    2. The control cost (magnitude of actions applied).

    Parameters:
        env: The original Gymnasium environment to wrap.
    """

    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        """
        Steps through the environment with the given action and computes a 2D reward.

        Args:
            action: The action to apply to the environment.

        Returns:
            observation: The next observation after applying the action.
            reward: A tuple containing (forward_velocity, control_cost).
            terminated: Whether the episode has terminated.
            truncated: Whether the episode has been truncated.
            info: Additional information from the environment.
        """
        # Perform a step in the environment
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Extract forward velocity (usually available in the info dictionary or state)
        forward_velocity = info["reward_forward"]  # Adjust based on your env specifics

        # Compute control cost as the sum of the squared actions
        control_cost = info["reward_ctrl"]

        # Create the 2-dimensional reward
        two_dim_reward = (forward_velocity, control_cost)

        return obs, two_dim_reward, terminated, truncated, info
