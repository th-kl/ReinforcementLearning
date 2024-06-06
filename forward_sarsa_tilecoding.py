#############################################################################################################
#                                                                                                           #
#   Implementation of the Forward Sarsa Algorithm as proposed by Van Seijen (2016):                         #
#    - Title: "Effective Multi-step Temporal-Difference Learning for Non-Linear Function Approximation"     #
#    - URL: <https://arxiv.org/abs/1608.05151>                                                              #
#                                                                                                           #
#############################################################################################################

import sys 
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import numpy as np
from Agents import MultiOutput_Discrete_SarsaAgent, Config
from Environments import CartPole, MountainCar
import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras import backend
import pandas as pd
from helpers import avg_pairwise_interference, compute_row_ratios
import json
import datetime
import argparse

# Manually compute and apply gradients for update
# tf.function wrapper used to improve speed and memory usage
@tf.function
def _compute_gradients(model, s_t_update, a_t_update):
    with tf.GradientTape() as tape:
        q_pred = model(s_t_update)[0][a_t_update]
    grads = tape.gradient(q_pred, model.trainable_weights)
    return grads, q_pred

@tf.function
def _apply_gradients(optimizer, model, scaled_grads):
    optimizer.apply_gradients(zip(scaled_grads, model.trainable_weights))

def discretized_forward_sarsa(params):
    config = Config(environment=CartPole(max_episode_steps=params['max_episode_steps']), 
                    alpha=2**(-params['alpha']),
                    gamma=params['gamma'],
                    episodes=params['episodes'],
                    epsilon_decay=params['epsilon_decay'],
                    min_epsilon=params['min_epsilon'])

    agent = MultiOutput_Discrete_SarsaAgent(config=config, 
                                            lamda=params['lamda'], 
                                            eta=params['eta'], 
                                            discretization=params['discretization'],
                                            n_tiles=params['n_tiles'],
                                            n_tilings=params['n_tilings'])
    env = agent.environment.env
    if params['optimizer'] == "SGD":
        optimizer = SGD
    elif params['optimizer'] == 'adam':
        optimizer = Adam
    else:
        raise ValueError("Unknown Optimizer. Please use: 'SGD' or 'adam'.")

    # Initialize model
    agent.create_model(optimizer=optimizer, 
                    loss=MeanSquaredError(), 
                    hidden_layers=params['hidden_layers'], 
                    activation=params['activation'])
    print(agent.model.summary())
    pairwise_interference = []
    row_ratios = []
    sample_df = pd.read_csv("sample_history.csv", index_col=0)
    samples = sample_df.to_numpy()
    samples = tf.convert_to_tensor(samples, dtype=tf.float32)
    del sample_df

    # Create batch of tile-coded tensors to evaluate
    discretized_samples = np.zeros((500, agent.feature_len))
    for sample_nr, sample in enumerate(samples):
        discretized_samples[sample_nr] = agent.create_feature(sample)
    
    # Compute PI and RR at initialization
    pairwise_interference.append(str(avg_pairwise_interference(agent.model, discretized_samples).numpy()))
    row_ratios.append(str(np.mean(compute_row_ratios(agent.model, discretized_samples))))
    
    F = deque(maxlen=agent.K)  
    c_final = (agent.gamma*agent.lamda)**(agent.K-1)

    # Please refer to the pseudo-code in the original paper for an explanation of the mechanics
    for episode in range(agent.episodes):
        if agent.iht.overfullCount > 0:
            raise ValueError(f"WARNING: OVERFULL COUNT -> {agent.iht.overfullCount}")

        U_sync = 0
        q_t = 0
        ready = False
        i = 0  
        c = 1  

        agent.epsilon = agent.epsilon_schedule(episode)
        s_t = agent.create_feature(env.reset()[0])
        a_t, _ = agent.epsilon_greedy_action(s_t) 
        terminated, truncated = False, False
        agent.score = 0
        agent.episode_loss = []

        while not (terminated or truncated):
            s_t1, r_t1, terminated, truncated, info = env.step(a_t)
            s_t1 = agent.create_feature(s_t1)
            a_t1, q_t1 = agent.epsilon_greedy_action(s_t1)
            agent.score += r_t1
            
            if terminated:
                q_t1 = 0
            
            rho = r_t1 + agent.gamma*(1-agent.lamda)*q_t1
            F.append((s_t, a_t, rho))
            delta = r_t1 + agent.gamma * q_t1 - q_t
            q_t = q_t1
            
            if i == agent.K - 1: 
                U = U_sync
                U_sync = q_t
                i = 0
                c = 1
                ready = True
            else:
                U_sync = U_sync + c * delta
                c = agent.gamma * agent.lamda * c
                i += 1
            
            if ready:
                U = U + c_final * delta
                s_t_update, a_t_update, rho_update = F.popleft()
                s_t_update = agent.create_input_tensor(s_t_update)
                grads, q_pred = _compute_gradients(agent.model, s_t_update, a_t_update)
                pred_error = U - q_pred
                agent.episode_loss.append(pred_error**2)
                scaled_grads = [(-1*pred_error) * grad for grad in grads]
                _apply_gradients(agent.optimizer, agent.model, scaled_grads)

                if agent.K != 1:
                    U = (U - rho_update) / (agent.gamma * agent.lamda)
                
            s_t = s_t1
            a_t = a_t1
            
        if ready == False:
            U = U_sync
        
        # Post-termination updates
        while len(F) != 0:
            s_t_update, a_t_update, rho_update = F.popleft()
            s_t_update = agent.create_input_tensor(s_t_update)
            grads, q_pred = _compute_gradients(agent.model, s_t_update, a_t_update)
            pred_error = U - q_pred
            agent.episode_loss.append(pred_error**2)
            scaled_grads = [(-1*pred_error) * grad for grad in grads]
            _apply_gradients(agent.optimizer, agent.model, scaled_grads)
            
            if agent.K != 1:
                U = (U - rho_update) / (agent.gamma * agent.lamda)

        if (((episode+1) in [1, 5, 10, 25]) or ((episode+1) % 25 == 0)):
            discretized_samples = np.zeros((500, agent.feature_len))
            for sample_nr, sample in enumerate(samples):
                discretized_samples[sample_nr] = agent.create_feature(sample)
            pairwise_interference.append(str(avg_pairwise_interference(agent.model, discretized_samples).numpy()))
            row_ratios.append(str(np.mean(compute_row_ratios(agent.model, discretized_samples))))


        agent.log_history(episode=episode, print_results=params['verbose'])
        
    # Write results
    results_dir = parent_dir+"/tilecoding_results"
    os.makedirs(results_dir, exist_ok=True)
    checkpoint_dict = {
        "scores": agent.score_history,
        'losses': agent.loss_history,
        'interference': pairwise_interference,
        'row_ratios': row_ratios,
        'params':params
    }
    results_filepath = os.path.join(results_dir, f"{datetime.datetime.now().strftime('%m%d_%H%M%S')}_{params['job_id']}_tilecoding_{params['trace']}_act_{params['activation']}_alpha_{str(params['alpha']).replace('.', '_')}_max_ep_{params['max_episode_steps']}_scaling_{params['observation_scaling']}_lamda_{str(params['lamda']).replace('.', '_')}_ntiles_{params['n_tiles']}_ntilings_{params['n_tilings']}.json")
    try:
        with open(results_filepath, "w") as json_file:
            json.dump(checkpoint_dict, json_file) 
        print(f"Results saved to {results_filepath}.")
    except:
        raise OSError("File could not be saved.")

    backend.clear_session()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-a', '--alpha', help='alpha', type=float)
    parser.add_argument('-g', '--gamma', help='gamma', type=float)
    parser.add_argument('-tr', '--trace', help='trace_type', type=str)
    parser.add_argument('-n_ep', '--n_episodes', help='n_episodes', type=int)
    parser.add_argument('-len_ep', '--len_episodes', help='max_episode_length', type=int)
    parser.add_argument('-act', '--activation', help='activation', type=str)
    parser.add_argument('-eta', help='eta_forward_sarsa', type=float)
    parser.add_argument('-o', '--optimizer', help='optimizer', type=str)
    parser.add_argument('-e_decay', '--epsilon_decay', help='epsilon_decay', type=float)
    parser.add_argument('-min_e', '--min_epsilon', help='min_epsilon', type=float)
    parser.add_argument('-l', '--lamda', help='lamda', type=float)
    parser.add_argument('-sc', '--scaling', help='scaling', type=str)
    parser.add_argument('-discr', '--discretization', help='discretization_method', type=str)
    parser.add_argument('-n_tiles', help='number_of_tiles', type=int)
    parser.add_argument('-n_tilings', help='number_of_tilings', type=int)
    parser.add_argument('-job_id', help='job_id', type=str)
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-visual', '--visualize', action='store_true')
    
    args = parser.parse_args()
    
    params={
        'trace':args.trace,
        'gamma':args.gamma,
        'episodes':args.n_episodes,
        'hidden_layers':[50],
        'activation':args.activation,
        'eta':args.eta,
        'max_episode_steps':args.len_episodes,
        'optimizer':args.optimizer,
        "alpha": args.alpha,
        "epsilon_decay": args.epsilon_decay,
        "min_epsilon": args.min_epsilon,
        "lamda": args.lamda,
        'observation_scaling':args.scaling,
        'discretization':args.discretization,
        'n_tiles':args.n_tiles,
        'n_tilings':args.n_tilings,
        'job_id':args.job_id,
        'verbose':args.verbose,
        'visual':args.visualize
    }

    try:
        discretized_forward_sarsa(params)
    except:
        raise ValueError(f"Error encountered with params: {params}")