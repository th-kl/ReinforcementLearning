import gymnasium as gym
import random
import numpy as np
import pandas as pd
from collections import deque
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod
from Tiles import IHT, tiles, tile_code_state
from typing import TypedDict
from Environments import Environment
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.initializers import Orthogonal, RandomUniform, GlorotUniform, Zeros, RandomNormal

class Config(TypedDict):
    environment: Environment
    alpha: float
    gamma: float
    episodes: int
    epsilon_decay: float
    min_epsilon: float
    max_episode_steps: int

# Base class for all agents
class Agent(ABC):
    def __init__(self, environment, episodes, alpha, gamma, epsilon_decay, min_epsilon, max_episode_steps=None):
        self.episodes = episodes
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_decay = epsilon_decay
        self.score = 0
        self.score_history = []
        self.episode_loss = []
        self.loss_history = []
        self.environment = environment
        self.value_estimates = []
        self.min_epsilon = min_epsilon

    @abstractmethod
    def create_input_tensor(self, state, action):
        pass
    
    @abstractmethod
    def greedy_action(self, state):
        pass

    @abstractmethod
    def predict_q_value(self, state, action):
        pass

    def epsilon_greedy_action(self, state):
        if np.random.rand() > self.epsilon:
            a_t, q_t = self.greedy_action(state=state)
        else:
            a_t = random.choice(self.environment.actions)
            q_t = self.predict_q_value(state=state, action=a_t)
        return a_t, q_t

    def epsilon_schedule(self, episode):
        epsilon = max(self.min_epsilon, self.epsilon_decay ** (episode+1))
        return epsilon

    def log_history(self, episode, print_results=False):
        self.episode = episode
        self.rmse = int(np.sqrt(np.mean(self.episode_loss)).astype(int))
        self.loss_history.append(self.rmse)
        self.score_history.append(int(self.score))

        if print_results:
            print(f"episode: {episode} score: {self.score} loss: {self.rmse}\n--------------------------------------------")

# Pure NumPy Implementation of a linear agent
class LinearSarsaAgent(Agent):
    def __init__(self, config:Config, lamda, n_tiles, n_tilings):
        super().__init__(environment=config['environment'], 
                         episodes=config['episodes'], 
                         alpha=config['alpha'], 
                         gamma=config['gamma'], 
                         epsilon_decay=config['epsilon_decay'], 
                         min_epsilon=config['min_epsilon'])
        self.lamda = lamda
        self.n_tiles = n_tiles
        self.n_tilings = n_tilings
        self.feature_len = ((self.n_tiles+1) ** self.environment.n_dim) * self.n_tilings
        self.iht = IHT(self.feature_len) # Create the index hash table
        self.params = {'alpha':self.alpha,
                       'gamma':self.gamma, 
                       'lambda':self.lamda,
                       'tiles_per_dim':self.n_tiles,
                       'tilings':self.n_tilings}

        self.weights = 0.01 * np.random.random_sample(size=self.feature_len*self.environment.n_actions) + (-0.01)

    def create_feature(self, state):
        # Tiles have unit-dimension -> multiply by n_tiles
        scaled_state = self.n_tiles * (self.environment.scale(state) / 2)
        return tile_code_state(scaled_state, self.n_tilings, self.iht, self.feature_len)

    def create_input_tensor(self, state, action):
        tensor = np.insert(np.zeros((self.environment.n_actions-1)*self.feature_len), action*self.feature_len, state)
        return tensor
    
    def greedy_action(self, state):
        q_values = []
        for action in self.environment.actions:
            s_t_a_t = self.create_input_tensor(state, action)
            q = self.weights @ s_t_a_t
            q_values.append(q)
        
        a_max, q_max = np.argmax(q_values), np.max(q_values)
        return a_max, q_max

    def predict_q_value(self, state, action):
        s_t_a_t = self.create_input_tensor(state, action)
        q = self.weights @ s_t_a_t
        return q
    
    def reset_trace(self):
        self.trace = np.zeros_like(self.weights)

# Tensorflow implementation of linear agent
class LinearSarsaAgent_V2(LinearSarsaAgent):
    def __init__(self, config, lamda, n_tiles, n_tilings, eta=0, max_K=50):
        super().__init__(config, lamda, n_tiles, n_tilings)
        self.eta = eta   
        if ((self.eta == None) or (self.eta == 0)):
            self.K = None
        elif self.lamda == 0:
            self.K = 1
        elif self.gamma * self.lamda == 1:
            self.K = max_K
        elif int(np.ceil(np.log(self.eta) / np.log(self.gamma * self.lamda))) > max_K:
                self.K = max_K
        else:
            self.K = int(np.ceil(np.log(self.eta) / np.log(self.gamma * self.lamda))) 
    
    def create_model(self, optimizer, loss=MeanSquaredError(), kernel_initializer=None, hidden_layers=[], activation='relu', weight_reg=None):
        self.loss_fn = loss
        self.model = Sequential()
        self.model.add(Input(shape=(self.feature_len*self.environment.n_actions,)))
        self.model.add(Dense(1, activation="linear", name='output_layer'))
        self.model.compile(loss=loss, optimizer=optimizer(learning_rate=self.alpha))
        self.optimizer = optimizer(learning_rate=self.alpha)
        self.params['optimizer'] = optimizer().name

    def create_input_tensor(self, state, action):
        tensor = np.insert(np.zeros((self.environment.n_actions-1)*self.feature_len), action*self.feature_len, state)
        return tensor.reshape(1, -1)

    def predict_q_value(self, state, action):
        s_t_a_t = self.create_input_tensor(state, action)
        q = self.predict_q(s_t_a_t)
        return q

    @tf.function
    def predict_q(self, s_t_a_t):
        return self.model(s_t_a_t)[0]
    
    def greedy_action(self, state):
        q_values = []
        for action in self.environment.actions:
            q = self.predict_q_value(state=state, action=action)
            q_values.append(q)    
        a_max, q_max = np.argmax(q_values), np.max(q_values)
        return a_max, q_max
    
    def reset_trace(self):
        self.traces = [tf.zeros(weights.shape) for weights in self.model.trainable_weights]

# Separate NN for each action (single output node for action-value) as reported by Van Seijen (2016)
class MultiNetSarsaAgent(Agent):
    def __init__(self, config:Config, lamda, eta=0.01, max_K=50):
        super().__init__(environment=config['environment'], 
                         episodes=config['episodes'], 
                         alpha=config['alpha'], 
                         gamma=config['gamma'], 
                         epsilon_decay=config['epsilon_decay'], 
                         min_epsilon=config['min_epsilon'])
        self.lamda = lamda
        self.eta = eta  
        if ((self.eta == None) or (self.eta == 0)):
            self.K = None
        elif self.lamda == 0:
            self.K = 1
        elif self.gamma * self.lamda == 1:
            self.K = max_K
        elif int(np.ceil(np.log(self.eta) / np.log(self.gamma * self.lamda))) > max_K:
                self.K = max_K
        else:
            self.K = int(np.ceil(np.log(self.eta) / np.log(self.gamma * self.lamda))) 
        
        self.params = {'alpha':self.alpha,
                       'gamma':self.gamma,
                       'lambda':self.lamda,
                       'epsilon_decay':self.epsilon_decay,
                       'min_epsilon':self.min_epsilon,
                       'eta':self.eta,
                       'K':self.K}

    def create_action_models(self, optimizer, loss=MeanSquaredError(), hidden_layers=[50], activation='relu', weight_reg=None):
        self.models = {}
        self.optimizers = {}
        self.loss_fn = loss
        for action in self.environment.actions:
            self.models[action] = Sequential()
            self.models[action].add(Input(shape=(self.environment.n_dim,)))
            for layer_nr, size in enumerate(hidden_layers):
                self.models[action].add(Dense(size, activation=activation, kernel_regularizer=weight_reg, name=f'dense_{layer_nr}'))
            self.models[action].add(Dense(1, activation="linear", name='output_layer'))
            self.models[action].compile(loss=loss, optimizer=optimizer(learning_rate=self.alpha))
            self.optimizers[action] = optimizer(learning_rate=self.alpha)
            print(optimizer().name)

        self.params['optimizer'] = optimizer().name

    def create_feature(self, state, scaling='minmax'):
        scaled_state = self.environment.scale(state, scaling=scaling)
        return scaled_state.reshape(1, -1)
    
    # create state-action-pair tensors
    def create_input_tensor(self, state):
        return np.reshape(state, (1, self.environment.n_dim))

    # return greedy action and corresponding q_value
    def greedy_action(self, state):
        q_values = []
        for action in self.environment.actions:
            q = self.predict_q_value(state, action)
            q_values.append(q)
        a_t = np.argmax(q_values)
        q_max = q_values[a_t]
        return a_t, q_max

    def predict_q_value(self, state, action):
        q = self.predict_q(self.models[action], state.reshape(1, -1))
        return q
    
    def predict_q(self, model, state):
        return model(state)[0]
    
    def reset_trace(self):
        self.traces = []
        for action in self.environment.actions:
            self.traces.append([tf.zeros(weights.shape) for weights in self.models[action].trainable_weights])

# Single network with one output node for each action-value
class MultiOutput_Discrete_SarsaAgent(Agent):
    """discretization dict: 
    TileCoding: \{'method':'tile_coding', 'n_tiles':n_tiles, 'n_tilings':n_tilings\} or Discretized: \{'method':'discrete', 'n_bins':n_bins\}"""
    def __init__(self, config:Config, lamda:float, discretization:str, n_tiles:int, n_tilings:int, eta:float):
        super().__init__(environment=config['environment'], 
                         episodes=config['episodes'], 
                         alpha=config['alpha'], 
                         gamma=config['gamma'], 
                         epsilon_decay=config['epsilon_decay'], 
                         min_epsilon=config['min_epsilon'])
        self.lamda = lamda
        if discretization in ['tile_coding', 'tilecoding']:
            self.method = 'tile_coding'
            self.n_tiles = n_tiles
            self.n_tilings = n_tilings
            self.feature_len = (self.n_tiles+1) ** self.environment.n_dim * self.n_tilings
            self.iht = IHT(self.feature_len) # Create the index hash table
            self.n_bins = None
        elif discretization in ['discrete', 'discretized']:
            self.method = 'discrete'
            self.n_bins = n_tiles
            self.feature_len = self.n_bins * self.environment.n_dim
            self.n_tilings = None
            self.n_tiles = None
        
        self.eta = eta  
        max_K = 50
        if self.lamda == 0:
            self.K = 1
        elif self.gamma * self.lamda == 1:
            self.K = max_K
        elif int(np.ceil(np.log(self.eta) / np.log(self.gamma * self.lamda))) > max_K:
                self.K = max_K
        else:
            self.K = int(np.ceil(np.log(self.eta) / np.log(self.gamma * self.lamda))) 
    
    def create_model(self, optimizer, loss=MeanSquaredError(), hidden_layers=[50], activation='relu', weight_reg=None, visualize_hidden_layer=None):
        self.model = Sequential()
        self.model.add(Input(shape=(self.feature_len,), name='input_layer'))
        for layer_nr, size in enumerate(hidden_layers):
            self.model.add(Dense(size, activation=activation, kernel_regularizer=weight_reg, name='dense_1'))
        self.model.add(Dense(self.environment.n_actions, activation="linear", name='output_layer'))
        
        self.model.compile(loss=loss, optimizer=optimizer(learning_rate=self.alpha))
        self.loss_fn = loss
        self.optimizer = optimizer(learning_rate=self.alpha)
        
        if visualize_hidden_layer != None:
            self.visualize = True
            self.visualized_layer = visualize_hidden_layer
        else:
            self.visualize = False

        # NOTE: Only implemented for CartPole environment
        if (self.visualize and isinstance(self.visualized_layer, int)):
            self.visual_model = Sequential()
            self.visual_model.add(Input(shape=(self.feature_len,)))
            for size in hidden_layers:
                self.visual_model.add(Dense(size, activation=activation, kernel_regularizer=weight_reg))
            
            self.visual_model.set_weights(self.model.layers[visualize_hidden_layer].get_weights())
            self.visual_model.compile(loss=loss, optimizer=optimizer(learning_rate=self.alpha))
    
    # featurize state through tile-coding
    def create_feature(self, state):
        if self.method in ['tilecoding', 'tile_coding']:
            # Tiles have unit-dimension -> multiply by n_tiles; division by two to correct scaling from -1:1 to -0.5:0.5
            scaled_state = self.n_tiles * (self.environment.scale(state) / 2)
            return tile_code_state(scaled_state, num_tilings=self.n_tilings, iht=self.iht, max_size=self.feature_len)
        elif self.method in ['discrete', 'discretized']:
            return self.environment.discretize(state, n_bins=self.n_bins)

    
    # create state-action-pair tensors
    def create_input_tensor(self, state):
        tensor = np.reshape(state, (1, self.feature_len))
        return tensor

    # return greedy action and corresponding q_value
    def greedy_action(self, state):
        tensor = self.create_input_tensor(state)
        q_values = self.model(tensor)
        a_t = np.argmax(q_values)
        q_max = np.max(q_values)
        return a_t, q_max

    def predict_q_value(self, state, action):
        tensor = self.create_input_tensor(state)
        q_values = self.model(tensor)[0][action]
        return q_values

    def update_visual_model(self):
        self.visual_model.set_weights(self.model.layers[self.visualized_layer].get_weights())

    def visualize_neuron_activity(self, interval, number, visualize_all_at_end=False):
        n_neurons = self.model.layers[0].units
        bins = self.environment.get_bins_for_visualization()
        n_bins = 16
        if (visualize_all_at_end and (self.episode >= self.episodes)):
            self.neurons = [idx for idx in range(n_neurons)]
        try:
            self.neurons
        except:
            self.neurons = np.random.choice(np.arange(n_neurons), number)

        if (self.episode + 1) % interval == 0:
            for neuron in self.neurons:
                for action in self.actions:
                    fig, ax = plt.subplots(5, 5, figsize=(16, 16))
                    for num_pos, position in enumerate(bins['position']):
                        for num_vel, velocity in enumerate(bins['velocity']):
                            state_grid = np.array(np.meshgrid(
                                bins['angle'],
                                bins['angular_velocity']
                            )).T.reshape(-1, 2)
                            state_grid = np.insert(state_grid, 0, position, axis=1)
                            state_grid = np.insert(state_grid, 1, velocity, axis=1)

                            input_tensors = np.zeros((256, self.feature_len+1))
                            for tensor_idx, tensor in enumerate(state_grid):
                                input_tensors[tensor_idx] = np.append(self.create_feature(tensor))

                            # Predict and reshape activations
                            activations = self.visual_model.predict(input_tensors, verbose=0)[:, neuron] 
                            activations = np.reshape(activations, (16, 16))

                            # Plot activations
                            im = ax[num_pos, num_vel].imshow(activations, cmap='hot', interpolation='nearest', vmin=np.min(activations), vmax=np.max(activations))
                            ax[num_pos, num_vel].set_xlabel("Angular Velocity Bins")
                            ax[num_pos, num_vel].set_ylabel("Angle Bins")
                            ax[num_pos, num_vel].label_outer()
                            ax[num_pos, num_vel].set_title(f"Pos: {position:.1f}, Vel: {velocity:.1f}, Act: {action}")

                    # Adjust layout for colorbar
                    fig.subplots_adjust(right=0.8)
                    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
                    fig.colorbar(im, cax=cbar_ax)
                    cbar_ax.set_label('Activation Intensity')

                    fig.suptitle(f'Neuron Activation Across State Space - Action {action}')
                    plt.savefig(f"neuron_{neuron}_activity_action_{action}_episode_{self.episode+1}")
                    plt.close()