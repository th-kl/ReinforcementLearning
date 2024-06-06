import numpy as np
import gymnasium as gym

class Environment:
    pass

class CartPole(Environment):
    def __init__(self, render_mode=None, max_episode_steps=500):
        self.env = gym.make('CartPole-v1', render_mode=render_mode, max_episode_steps=max_episode_steps)
        self.n_actions = self.env.action_space.n
        self.actions = np.arange(self.n_actions)
        self.n_dim = self.env.observation_space.shape[0]
        self.angular_vel_bins = np.linspace(-3.3, 3.3, 16 + 1)
        self.angle_bins = np.linspace(-0.21, 0.21, 16 + 1)
        self.max_cart_position = 2.4
        self.max_cart_velocity = 2.4
        self.max_pole_angle = 0.21
        self.max_angular_velocity = 2.75
    
    def scale(self, state, scaling='minmax'):
        min_max_factors = np.array([self.max_cart_position, 
                                    self.max_cart_velocity, 
                                    self.max_pole_angle, 
                                    self.max_angular_velocity])
        standard_factors = np.array([0.79,
                                     0.45,
                                     0.041,
                                     0.29])
        if scaling == 'minmax':
            return state / min_max_factors
        if scaling == 'normalized':
            return state / standard_factors
        else:
            raise ValueError("Unknown Transform")

    def get_bins_for_visualization(self):
        n_bins = 5
        bins = {
            'position': np.linspace(-1, 1, 5),
            'velocity': np.linspace(-1, 1, 5),
            'angle': np.linspace(-self.max_pole_angle, self.max_pole_angle, n_bins),
            'angular_velocity': np.linspace(-self.max_angular_velocity, self.max_angular_velocity, n_bins),
        }
    
    def discretize(self, state, n_bins, scaling=None):
        bins = np.array([np.linspace(-self.max_cart_position, self.max_cart_position, n_bins-2), 
                            np.linspace(-self.max_cart_velocity, self.max_cart_velocity, n_bins-2), 
                            np.linspace(-self.max_pole_angle, self.max_pole_angle, n_bins-2), 
                            np.linspace(-self.max_angular_velocity, self.max_angular_velocity, n_bins-2)
                            ])
        idx = np.zeros(self.n_dim).astype(int)
        for i, (var, b) in enumerate(zip(state, bins)):
            idx[i] = np.digitize(var, b)

        one_hot_state = np.zeros((self.n_dim, n_bins))
        one_hot_state[np.arange(idx.size), idx] = 1
        return one_hot_state.flatten()
    
class MountainCar(Environment):
    def __init__(self, render_mode=None, max_episode_steps=200):
        self.env = gym.make('MountainCar-v0', render_mode=render_mode, max_episode_steps=max_episode_steps)
        self.n_actions = self.env.action_space.n
        self.actions = np.arange(self.n_actions)
        self.n_dim = self.env.observation_space.shape[0]
        self.min_cart_position = -1.2 
        self.max_cart_position = 0.6
        self.max_cart_velocity = 0.07

    def scale(self, state, scaling='minmax'):
        scaled_state = np.array([
            ((2 * state[0]) / (1.8)) + 1/3,
            state[1] / (self.max_cart_velocity)
        ])
        return scaled_state

    def scale_state(self, state):
        scaled_state = np.array([
            state[0] / (-1*self.min_cart_position + self.max_cart_position),
            state[1] / (2 * self.max_cart_velocity)
        ])
        return scaled_state
    

    def get_bins_for_visualization(self):
        n_bins = 20
        bins = {
            'position': np.linspace(self.min_cart_position, self.max_cart_position, n_bins),
            'velocity': np.linspace(-self.max_cart_velocity, self.max_cart_velocity, n_bins)
        }
        return bins