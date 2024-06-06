#####################################################################################
#                                                                                   #
#   Implementation of Sarsa with Accumulating Traces as proposed by Sutton (1988)   #
#    - Title: "Learning to predict by the methods of temporal differences."         #
#    - URL: <https://doi.org/10.1023/A:1022633531479>                               #
#                                                                                   #
#####################################################################################

import sys 
import os
from pathlib import Path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)
import numpy as np
from Agents import MultiNetSarsaAgent, Config
from Environments import CartPole, MountainCar
import tensorflow as tf
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras import backend
import json
import gymnasium as gym
import pandas as pd
from helpers import avg_pairwise_interference, compute_row_ratios
import datetime
import argparse 

# Manually compute and apply gradients for update
# tf.function wrapper used to improve speed and memory usage
@tf.function
def _compute_gradients(model, state):
    with tf.GradientTape() as tape:
        q_pred = model(state)[0]
    grads = tape.gradient(q_pred, model.trainable_weights)
    return grads, q_pred

@tf.function
def _apply_gradients(optimizer, model, scaled_grads):
    optimizer.apply_gradients(zip(scaled_grads, model.trainable_weights))


def sarsa_multinet(params):

    config = Config(environment=CartPole(max_episode_steps=params['max_episode_steps']), 
                    alpha=2**(-params['alpha']),
                    gamma=params['gamma'],
                    episodes=params['episodes'],
                    epsilon_decay=params['epsilon_decay'],
                    min_epsilon=params['min_epsilon'])

    agent = MultiNetSarsaAgent(config=config, lamda=params['lamda'], eta=params['eta'])
    env = agent.environment.env

    if params['optimizer'] == "SGD":
        optimizer = SGD
    elif params['optimizer'] == 'adam':
        optimizer = Adam
    else:
        raise ValueError('Unknown optimizer. Please use: "adam" or "SGD".')
    
    agent.create_action_models(optimizer=optimizer, 
                    loss=MeanSquaredError(), 
                    hidden_layers=params['hidden_layers'], 
                    activation=params['activation']
                    )

    sample_df = pd.read_csv("sample_history.csv", index_col=0)
    samples = sample_df.to_numpy()
    samples = tf.convert_to_tensor(samples, dtype=tf.float32)
    del sample_df
    pairwise_interference = []
    row_ratios = []
    scaled_samples = agent.environment.scale(samples, scaling=params['observation_scaling'])
    pairwise_interference.append(str(avg_pairwise_interference(agent.models[0], scaled_samples).numpy()))
    row_ratios.append(str(np.mean(compute_row_ratios(agent.models[0], scaled_samples))))
    
    # Warm-up for optimizer to enable use of @tf.function in order to improve performance and memory usage
    for key, model in agent.models.items():
        optimizer = agent.optimizers[key]
        if hasattr(model, 'input_shape'):
            dummy_input = tf.random.normal([1, 4])
            with tf.GradientTape() as tape:
                dummy_output = model(dummy_input)
                dummy_loss = tf.reduce_mean(dummy_output)

            dummy_grads = tape.gradient(dummy_loss, model.trainable_weights) # Compute gradients
            optimizer.apply_gradients(zip(dummy_grads, model.trainable_weights)) # Apply gradients to initialize the optimizer's internal variables

    # Training loop                            
    for episode in range(agent.episodes):
        agent.epsilon = agent.epsilon_schedule(episode)
        s_t = env.reset()[0]
        s_t = agent.create_feature(s_t) 
        a_t, q_t = agent.epsilon_greedy_action(s_t)
        terminated, truncated = False, False
        agent.score = 0
        agent.reset_trace()
        agent.episode_loss = []
    
        while not (terminated or truncated):
            s_t1, r_t1, terminated, truncated, info = env.step(a_t)
            s_t1 = agent.create_feature(s_t1)
            
            agent.score += r_t1
            
            a_t1, q_t1 = agent.epsilon_greedy_action(s_t1)

            if terminated: # if goal state reached (terminated -> goal; truncated -> manually stopped)
                q_t1 = tf.zeros(1,)
            
            q_target = r_t1 + agent.gamma * q_t1

            # Compute gradients
            grads, q_pred = _compute_gradients(agent.models[a_t], s_t)
            
            
            td_error = q_target.numpy() - q_pred.numpy()
            for i, grad in enumerate(grads):
                if params['trace'] == 'accumulating': 
                    agent.traces[a_t][i] = agent.gamma * agent.lamda * agent.traces[a_t][i] + grad
                else:
                    raise ValueError("Please specify trace: 'accumulating'.")

            # Apply update
            scaled_grads = [(-1*td_error) * trace for trace in agent.traces[a_t]]
            _apply_gradients(agent.optimizers[a_t], agent.models[a_t], scaled_grads)
            
            
            s_t = s_t1
            a_t = a_t1
            agent.episode_loss.append(float(td_error)**2)

        # Compute pairwise interference and row ratio on kernel over sampled states
        if (((episode+1) in [1, 5, 10, 25]) or ((episode+1) % 25 == 0)):
            scaled_samples = agent.environment.scale(samples, scaling=params['observation_scaling'])
            pairwise_interference.append(str(avg_pairwise_interference(agent.models[0], scaled_samples).numpy()))
            row_ratios.append(str(np.mean(compute_row_ratios(agent.models[0], scaled_samples))))
        
        agent.log_history(episode=episode, print_results=params['verbose'])
    
    results_dir = parent_dir+"/sarsa_results"
    os.makedirs(results_dir, exist_ok=True)
    checkpoint_dict = {
        "scores": agent.score_history,
        'losses': agent.loss_history,
        'interference': pairwise_interference,
        'row_ratios': row_ratios
    }
    results_filepath = os.path.join(results_dir, f"{datetime.datetime.now().strftime('%m%d_%H%M%S')}_{params['job_id']}_sarsa_{params['trace']}_act_{params['activation']}_alpha_{str(params['alpha']).replace('.', '_')}_max_ep_{params['max_episode_steps']}_scaling_{params['observation_scaling']}_lamda_{str(params['lamda']).replace('.', '_')}_ntiles_{params['n_tiles']}_ntilings_{params['n_tilings']}.json")
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
    parser.add_argument('-n_tiles', help='number_of_tiles', type=int)
    parser.add_argument('-n_tilings', help='number_of_tilings', type=int)
    parser.add_argument('-job_id', help='job_id', type=str)
    parser.add_argument('-v', '--verbose', action='store_true')
    
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
        'n_tiles':args.n_tiles,
        'n_tilings':args.n_tilings,
        'job_id':args.job_id,
        'verbose':args.verbose
    }

    try:
        sarsa_multinet(params)
    except:
        raise ValueError(f"Error encountered with params: {params}")