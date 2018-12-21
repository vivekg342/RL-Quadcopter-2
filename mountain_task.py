import numpy as np
import gym

class Task():
    """Sample task of OpenGym Mountain Car"""
    def __init__(self, seed=503):
        """Initialize a Mountain Car Task object.
        """
        # Simulation
        self.env = gym.make('MountainCar-v0')
        self.state_size = env.observation_space.shape[0]
        self.action_low = env.observation_space.action_low
        self.action_high = env.observation_space.action_low
        self.action_size = env.action_space.n
        self.state = self.env.reset()

    def step(self, rotor_speeds):
        """Uses action to obtain next state, reward, done."""
        reward = 0
        state, reward, done, _ = env.step(action)
        next_state = np.concatenate(pose_all)
        return next_state, reward, done

    def reset(self):
        """Reset  to start a new episode."""
        return self.env.reset()