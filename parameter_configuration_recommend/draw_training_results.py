import sys
sys.path.append('../')

import os
import torch
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Args import args as args_r
from query_performance_predict.Args import args as args_p

def draw_reward(episode_score_file, sava_path):
    with open(episode_score_file, 'rb') as f:
        episode_score = pickle.load(f)

    avg_rewards = []
    num_episodes = len(episode_score)
    group_size = 10

    episodes = list(episode_score.keys())
    rewards = list(episode_score.values())
    
    num = 0
    for i in range(0, num_episodes, group_size):
        avg_reward = np.mean(rewards[i:i + group_size])
        if avg_reward > -10000:
            avg_rewards.append(avg_reward)
            num = num + 1

    plt.figure(figsize=(24, 12))

    plt.subplot(1, 2, 1)
    plt.plot(episodes, rewards, linestyle='-', color='b')

    plt.title('Total Rewards per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(range(0, num*group_size, group_size), avg_rewards, linestyle='-', color='r')

    plt.title('Average Total Rewards per 10 Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Average Total Reward')
    plt.grid(True)

    plt.savefig(sava_path)

def draw_steps(episode_steps_file, sava_path):
    with open(episode_steps_file, 'rb') as f:
        episode_steps = pickle.load(f)

    episodes = list(episode_steps.keys())
    steps = list(episode_steps.values())

    plt.figure(figsize=(24, 12))
   
    plt.plot(episodes, steps, linestyle='-', color='b')

    plt.title('Total Steps per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Total Stpeps')
    plt.grid(True)

    plt.savefig(sava_path)

def draw_loss(episode_closs_file, episode_aloss_file, sava_path):
    with open(episode_closs_file, 'rb') as f:
        episode_closs = pickle.load(f)

    with open(episode_aloss_file, 'rb') as f:
        episode_aloss = pickle.load(f)

    episodes = list(episode_closs.keys())
    closs = list(episode_closs.values())
    aloss = list(episode_aloss.values())
    
    plt.figure(figsize=(24, 12))

    plt.subplot(1, 2, 1)
    plt.plot(episodes, closs, linestyle='-', color='b')

    plt.title('Critic Loss per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Critic Loss')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(episodes, aloss, linestyle='-', color='r')
    plt.title('Actor Loss per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Actor Loss')
    plt.grid(True)

    plt.savefig(sava_path)

if __name__ == '__main__':
    current_directory = os.getcwd()
    parent_directory = os.path.dirname(current_directory)

    store_dir = './{}_{}_TD3'.format(args_r.actor_layer_sizes, args_r.critic_layer_sizes)
    # store_dir = './{}_{}_TD3_nsg'.format(args_r.actor_layer_sizes, args_r.critic_layer_sizes)

    expr_name = '{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(args_r.epoches, args_r.max_steps, args_r.batch_size, args_r.alr,
                                                    args_r.tau, args_r.sigma, args_r.delay_time, args_r.pec_reward, args_r.nochange_steps)

    training_mode = args_r.training_mode

    if training_mode  == 'train':
        for name in ('episode_reward_fig', 'episode_steps_fig', 'episode_loss_fig'):
            os.makedirs(os.path.join(store_dir, name), exist_ok=True)
            
        episode_score_file = os.path.join(store_dir, 'train_episode_score_{}.pkl'.format(expr_name))
        episode_steps_file = os.path.join(store_dir, 'train_episode_steps_{}.pkl'.format(expr_name))
        episode_closs_file = os.path.join(store_dir, 'train_episode_closs_{}.pkl'.format(expr_name))
        episode_aloss_file = os.path.join(store_dir, 'train_episode_aloss_{}.pkl'.format(expr_name))

        episode_reawrd_fig_path = os.path.join(store_dir, 'episode_reward_fig/train_episode_reward_fig_{}.png'.format(expr_name))
        episode_steps_fig_path = os.path.join(store_dir, 'episode_steps_fig/train_episode_steps_fig_{}.png'.format(expr_name))
        episode_loss_fig_path = os.path.join(store_dir, 'episode_loss_fig/train_episode_loss_fig_{}.png'.format(expr_name))
    elif training_mode  == 'evaluate':
        experiment_mode = args_p.experiment_mode

        store_subdir = os.path.join(store_dir, '{}'.format(experiment_mode))
        os.makedirs(store_subdir, exist_ok=True)

        for name in ('episode_reward_fig', 'episode_steps_fig', 'episode_loss_fig'):
            os.makedirs(os.path.join(store_subdir, name), exist_ok=True)

        dataset_name = args_r.dataset_name

        episode_score_file = os.path.join(store_subdir, 'eval_episode_score_{}_{}_{}_{}.pkl'.format(expr_name, dataset_name, args_r.nochange_episodes, args_r.test_epoches))
        episode_steps_file = os.path.join(store_subdir, 'eval_episode_steps_{}_{}_{}_{}.pkl'.format(expr_name, dataset_name, args_r.nochange_episodes, args_r.test_epoches))
        episode_closs_file = os.path.join(store_subdir, 'eval_episode_closs_{}_{}_{}_{}.pkl'.format(expr_name, dataset_name, args_r.nochange_episodes, args_r.test_epoches))
        episode_aloss_file = os.path.join(store_subdir, 'eval_episode_aloss_{}_{}_{}_{}.pkl'.format(expr_name, dataset_name, args_r.nochange_episodes, args_r.test_epoches))


        episode_reawrd_fig_path = os.path.join(store_subdir,
                                               'episode_reward_fig/eval_episode_reward_fig_{}_{}_{}_{}.png'.format(expr_name, dataset_name, args_r.nochange_episodes, args_r.test_epoches))
        episode_steps_fig_path = os.path.join(store_subdir,
                                              'episode_steps_fig/eval_episode_steps_fig_{}_{}_{}_{}.png'.format(expr_name, dataset_name, args_r.nochange_episodes, args_r.test_epoches))
        episode_loss_fig_path = os.path.join(store_subdir,
                                             'episode_loss_fig/eval_episode_loss_fig_{}_{}_{}_{}.png'.format(expr_name, dataset_name, args_r.nochange_episodes, args_r.test_epoches))

    draw_reward(episode_score_file, episode_reawrd_fig_path)
    draw_steps(episode_steps_file, episode_steps_fig_path)
    draw_loss(episode_closs_file, episode_aloss_file, episode_loss_fig_path)







