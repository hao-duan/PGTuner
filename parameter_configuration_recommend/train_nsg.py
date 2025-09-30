# -*- coding: utf-8 -*-
"""
Train the model for the NSG index.
"""

import sys
sys.path.append('../')

import os
import pickle
import pandas as pd
import numpy as np
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter
import random
from tqdm import tqdm
import time

from query_performance_predict.Args import args as args_p
from query_performance_predict.utils import df2np
from TD3_nsg import *
from index_env import IndexEnv_nsg
from Args import args as args_r
from utils import Logger, time_start, time_end

if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)

    random.seed(args_r.seed)
    np.random.seed(args_r.seed)
    torch.manual_seed(args_r.seed)
    torch.cuda.manual_seed(args_r.seed)
    torch.cuda.manual_seed_all(args_r.seed)
    cudnn.enabled = False
    cudnn.benchmark = False
    cudnn.deterministic = True

    print('----------------preprocess----------------')
    use_cuda = torch.cuda.is_available()
    print(use_cuda)
    device = torch.device("cuda" if use_cuda else "cpu")

    current_directory = os.getcwd()
    parent_directory = os.path.dirname(current_directory)

    predict_model_save_path = os.path.join(parent_directory, 'query_performance_predict/model_checkpoints/NSG_KNNG/{}_{}_{}_{}_checkpoint.pth'.format(
                                               args_p.dipredict_layer_sizes_nsg, args_p.dipredict_n_epochs,
                                               args_p.dipredict_batch_size, args_p.dipredict_lr))
    standard_path = os.path.join(parent_directory,
                                 'query_performance_predict/scaler_paras/NSG_KNNG/feature_standard.npz')

    data_path = os.path.join(parent_directory, 'NSG_KNNG/Data/train_data.csv')

    df_ini = pd.read_csv(data_path, sep=',', header=0)
    def_df = df_ini[(df_ini['K'] == 100) & (df_ini['L'] == 100) & (df_ini['L_nsg_C'] == 150) & (df_ini['R_nsg'] == 5) & (df_ini['C'] == 300) & (df_ini['L_nsg_S'] == 10)]

    dataset_names = def_df['FileName']

    feature_df = def_df[['SIZE', 'q_SIZE', 'DIM', 'LID', 'Sum_K_MinDist', 'Sum_K_MaxDist', 'Sum_K_StdDist', 'q_K_MinRatio', 'q_K_MeanRatio', 'q_K_MaxRatio', 'q_K_StdRatio']]
    def_performance_df = def_df[['recall', 'average_NSG_s_dc_counts']]

    target_rec_lis=[0.85, 0.88, 0.9, 0.92, 0.94, 0.95, 0.96, 0.98, 0.99]
    num_target_rec = len(target_rec_lis)

    data_feature = df2np(feature_df)
    default_performance = df2np(def_performance_df)

    data_feature[:, 0:2] = np.log10(data_feature[:, 0:2])

    final_data_feature = np.repeat(data_feature, num_target_rec, axis=0)
    final_default_performance = np.repeat(default_performance, num_target_rec, axis=0)

    num_dataset = data_feature.shape[0]
    num_data = num_dataset * num_target_rec

    store_dir = './{}_{}_TD3_nsg'.format(args_r.actor_layer_sizes, args_r.critic_layer_sizes)
    os.makedirs(store_dir, exist_ok=True)

    for name in ('log', 'runs', 'save_memory', 'model_params'):
        os.makedirs(os.path.join(store_dir, name), exist_ok=True)

    expr_name = '{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(args_r.epoches, args_r.max_steps, args_r.batch_size, args_r.alr,
                                                    args_r.tau, args_r.sigma, args_r.delay_time, args_r.pec_reward, args_r.nochange_steps)

    best_performance_file = os.path.join(store_dir, 'train_best_performance_{}.pkl'.format(expr_name))
    best_paras_file = os.path.join(store_dir, 'train_best_paras_{}.pkl'.format(expr_name))

    episode_score_file = os.path.join(store_dir, 'train_episode_score_{}.pkl'.format(expr_name))
    episode_steps_file = os.path.join(store_dir, 'train_episode_steps_{}.pkl'.format(expr_name))
    episode_closs_file = os.path.join(store_dir, 'train_episode_closs_{}.pkl'.format(expr_name))
    episode_aloss_file = os.path.join(store_dir, 'train_episode_aloss_{}.pkl'.format(expr_name))

    logger = Logger(name=args_r.method, log_file=os.path.join(store_dir, 'log/train_{}.log'.format(expr_name)))
    memory_file = os.path.join(store_dir, 'save_memory/train_{}.pkl'.format(expr_name))

    # writer_dir = os.path.join(store_dir, 'runs/train_{}'.format(expr_name))
    # if not os.path.exists(writer_dir):
    #     os.mkdir(writer_dir)
    # writer = SummaryWriter(writer_dir)

    print('----------------initialize the environment----------------')
    env = IndexEnv_nsg(num_dataset, final_default_performance, target_rec_lis, args_r, args_p, predict_model_save_path, standard_path, device)

    print('----------------load the TD3 model----------------')
    ddpg_opt = dict()
    ddpg_opt['tau'] = args_r.tau
    ddpg_opt['alr'] = args_r.alr
    ddpg_opt['clr'] = args_r.clr

    gamma = 0.9

    ddpg_opt['gamma'] = gamma
    ddpg_opt['max_steps'] = args_r.max_steps
    ddpg_opt['batch_size'] = args_r.batch_size
    ddpg_opt['memory_size'] = args_r.memory_size

    ddpg_opt['sigma_decay_rate'] = args_r.sigma_decay_rate
    ddpg_opt['sigma'] = args_r.sigma
    ddpg_opt['delay_time'] = args_r.delay_time

    ddpg_opt['actor_layer_sizes'] = eval(args_r.actor_layer_sizes)
    ddpg_opt['critic_layer_sizes'] = eval(args_r.critic_layer_sizes)

    ddpg_opt['actor_path'] = os.path.join(store_dir, 'model_params/actor_{}.pth'.format(expr_name))
    ddpg_opt['critic1_path'] = os.path.join(store_dir, 'model_params/critic1_{}.pth'.format(expr_name))
    ddpg_opt['critic2_path'] = os.path.join(store_dir, 'model_params/critic2_{}.pth'.format(expr_name))

    n_states = args_r.n_states_nsg
    n_actions = args_r.n_actions_nsg

    model = TD3(n_states=n_states, n_actions=n_actions, num_data=num_data, opt=ddpg_opt, dv=device)

    episode_score = {}
    episode_steps = {}
    episode_closs = {}
    episode_aloss = {}

    if os.path.exists(memory_file):
        model.replay_memory.load_memory(memory_file)
        print("Load Memory: {}".format(len(model.replay_memory)))

    # the total time of one step is completed, including env_step_time, action_step_time, and train_step_time
    step_times = []
    # time for training
    train_step_times = []
    # time for the environment to execute one step
    env_step_times = []
    # time for adding experience
    add_step_times = []
    # time for recommending an action
    action_step_times = []

    # record the total reward of each episode
    total_scores = []

    print('----------------start training...----------------')
    start_time = time.time()
    for episode in tqdm(range(args_r.epoches), total=args_r.epoches):
        current_states = env._initialize()

        model.reset(args_r.sigma)

        train_step = 0
        accumulate_loss = [0, 0]

        for st in tqdm(range(args_r.max_steps), total=args_r.max_steps):
            step_time = time_start()
            states = current_states

            action_step_time = time_start()
            actions = model.choose_action(states, True)
            action_step_time = time_end(action_step_time)

            env_step_time = time_start()
            rewards, states_, dones, _, _, _, _ = env._step(actions, final_data_feature, best_performance_file, best_paras_file)  # filename是指最佳性能存储文件
            env_step_time = time_end(env_step_time)

            next_states = states_

            add_step_time = time_start()
            model.add_sample(states, actions, rewards, next_states, dones)
            add_step_time = time_end(add_step_time)

            current_states = next_states
            train_step_time = 0.0

            if len(model.replay_memory) > args_r.batch_size:
                losses = []
                train_step_time = time_start()
                for i in range(2):
                    loss = model.update()
                    if (model.update_time % model.delay_time) == 0:
                        losses.append(loss)
                        train_step += 1

                train_step_time = time_end(train_step_time) / 2

                accumulate_loss[0] += sum([x[0] for x in losses])
                accumulate_loss[1] += sum([x[1] for x in losses])

            step_time = time_end(step_time)
            step_times.append(step_time)

            env_step_times.append(env_step_time)
            add_step_times.append(add_step_time)
            train_step_times.append(train_step_time)
            action_step_times.append(action_step_time)

            if env.nochange_steps == args_r.nochange_steps or env.score < -2000:
                break

        model.actor_scheduler.step()
        model.critic1_scheduler.step()
        model.critic2_scheduler.step()

        episode_score[episode] = env.score
        episode_steps[episode] = env.steps
        episode_closs[episode] = accumulate_loss[0] / train_step
        episode_aloss[episode] = accumulate_loss[1] / train_step

        print(f'the score of current episode: {env.score}, the number of steps elapsed ：{env.steps}')
        if episode == 0:
            print(f'****************************{episode + 1} episode finishes****************************')
        else:
            print(f'****************************{episode + 1} episodes finish****************************')

        if (episode + 1) % 10 == 0:
            logger.info(
                "[Episode: {}][Average] step: {}s env step: {}s add step: {}s train step: {}s action time: {}s".format(
                    episode, np.mean(step_times),
                    np.mean(env_step_times), np.mean(add_step_times), np.mean(train_step_times),
                    np.mean(action_step_times)))

        if (episode + 1) % 10 == 0:
            model.replay_memory.save(memory_file)
            model.save_model(episode)

        if (episode + 1) % 10 == 0:
            with open(episode_score_file, 'wb') as f:
                pickle.dump(episode_score, f)
        if (episode + 1) % 10 == 0:
            with open(episode_steps_file, 'wb') as f:
                pickle.dump(episode_steps, f)
        if (episode + 1) % 10 == 0:
            with open(episode_closs_file, 'wb') as f:
                pickle.dump(episode_closs, f)
        if (episode + 1) % 10 == 0:
            with open(episode_aloss_file, 'wb') as f:
                pickle.dump(episode_aloss, f)

    end_time = time.time()
    training_time = end_time - start_time
    logger.info("Training time: {}s".format(training_time))





