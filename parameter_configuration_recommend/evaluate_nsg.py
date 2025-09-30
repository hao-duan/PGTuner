# -*- coding: utf-8 -*-

"""
Online recommendation for the NSG index
"""

import sys
sys.path.append('../')

import os
import pickle
import numpy as np
import pandas as pd
from torch.backends import cudnn
import random
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import time

from query_performance_predict.Args import args as args_p
from query_performance_predict.utils import df2np
from TD3_nsg import *
from index_env import IndexEnv_nsg
from Args import args as args_r
from utils import Logger

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

    filename_dic = {'deep1': 'deep_0_1_96_1', 'paper': 'paper_0_2_200_1', 'gist': 'gist_0_1_960_1'}

    target_rec_lis = [0.9, 0.95, 0.99]
    num_target_rec = len(target_rec_lis)

    # args.dataset_name = 'gist'
    dataset_name = args_r.dataset_name
    filename = filename_dic[dataset_name]

    max_selected_num = args_p.max_selected_num #args_p.max_selected_num = 12
    selected_num = 2
    selected_rounds = max_selected_num // selected_num

    mode = args_p.experiment_mode

    predict_model_save_path = os.path.join(parent_directory, 'query_performance_predict/model_checkpoints/NSG_KNNG/{}/{}_{}_{}_{}_{}_{}_{}_checkpoint.pth'.format(
                                      mode, dataset_name, args_p.dipredict_layer_sizes_nsg, args_p.dipredict_n_epochs,
                                           args_p.dipredict_batch_size, args_p.dipredict_lr, selected_num, selected_rounds))
    standard_path = os.path.join(parent_directory, 'query_performance_predict/scaler_paras/NSG_KNNG/{}/{}_feature_standard_{}_{}.npz'.format(
                                      mode, dataset_name, selected_num, selected_rounds))

    if mode == 'main' or mode == 'dataset_change':
        data_path = os.path.join(parent_directory, 'NSG_KNNG/Data/test_data_main.csv')
    elif mode == 'ds_change':
        data_path = os.path.join(parent_directory, 'NSG_KNNG/Data/test_data_ds_change.csv')
    elif mode == 'qd_change':
        data_path = os.path.join(parent_directory, 'NSG_KNNG/Data/test_data_qd_change.csv')

    df = pd.read_csv(data_path, sep=',', header=0)

    df_ini = df[df['FileName']==filename]
    def_df = df_ini[(df_ini['K'] == 100) & (df_ini['L'] == 100) & (df_ini['L_nsg_C'] == 150) & (df_ini['R_nsg'] == 5) & (df_ini['C'] == 300) & (df_ini['L_nsg_S'] == 10)]

    feature_df = def_df[['SIZE', 'q_SIZE', 'DIM', 'LID', 'Sum_K_MinDist', 'Sum_K_MaxDist', 'Sum_K_StdDist', 'q_K_MinRatio', 'q_K_MeanRatio', 'q_K_MaxRatio', 'q_K_StdRatio']]
    def_performance_df = def_df[['recall', 'average_NSG_s_dc_counts']]

    data_feature = df2np(feature_df)
    default_performance = df2np(def_performance_df)

    data_feature[:, 0:2] = np.log10(data_feature[:, 0:2])

    final_data_feature = np.repeat(data_feature, num_target_rec, axis=0)
    final_default_performance = np.repeat(default_performance, num_target_rec, axis=0)

    num_dataset = data_feature.shape[0]
    num_data = num_dataset * num_target_rec

    store_dir = './{}_{}_TD3_nsg'.format(args_r.actor_layer_sizes, args_r.critic_layer_sizes)
    os.makedirs(store_dir, exist_ok=True)

    store_subdir = os.path.join(store_dir, '{}'.format(mode))
    os.makedirs(store_subdir, exist_ok=True)

    for name in ('log', 'runs', 'save_memory', 'model_params'):
        os.makedirs(os.path.join(store_subdir, name), exist_ok=True)

    expr_name = '{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(args_r.epoches, args_r.max_steps, args_r.batch_size, args_r.alr, args_r.tau,
                                                                         args_r.sigma, args_r.delay_time, args_r.pec_reward, args_r.nochange_steps)

    best_performance_file = os.path.join(store_subdir,
                                         'eval_best_performance_{}_{}_{}_{}.pkl'.format(expr_name, dataset_name, args_r.nochange_episodes, args_r.test_epoches))
    best_paras_file = os.path.join(store_subdir,
                                   'eval_best_paras_{}_{}_{}_{}.pkl'.format(expr_name, dataset_name, args_r.nochange_episodes, args_r.test_epoches))

    episode_score_file = os.path.join(store_subdir, 'eval_episode_score_{}_{}_{}_{}.pkl'.format(expr_name, dataset_name, args_r.nochange_episodes, args_r.test_epoches))
    episode_steps_file = os.path.join(store_subdir, 'eval_episode_steps_{}_{}_{}_{}.pkl'.format(expr_name, dataset_name, args_r.nochange_episodes, args_r.test_epoches))
    episode_closs_file = os.path.join(store_subdir, 'eval_episode_closs_{}_{}_{}_{}.pkl'.format(expr_name, dataset_name, args_r.nochange_episodes, args_r.test_epoches))
    episode_aloss_file = os.path.join(store_subdir, 'eval_episode_aloss_{}_{}_{}_{}.pkl'.format(expr_name, dataset_name, args_r.nochange_episodes, args_r.test_epoches))

    logger = Logger(name=args_r.method, log_file=os.path.join(store_subdir, 'log/eval_{}_{}_{}_{}.log'.format(expr_name, dataset_name, args_r.nochange_episodes, args_r.test_epoches)))
    memory_file = os.path.join(store_subdir, 'save_memory/train_{}.pkl'.format(expr_name))

    print('----------------initialize the environment----------------')
    env = IndexEnv_nsg(num_dataset, final_default_performance, target_rec_lis, args_r, args_p,
                   predict_model_save_path, standard_path, device)

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

    model = TD3(n_states=n_states, n_actions=n_actions, num_data=num_target_rec, opt=ddpg_opt, dv=device)

    episode_score = {}
    episode_steps = {}
    episode_closs = {}
    episode_aloss = {}

    # record the total reward of each episode
    total_scores = []

    start_time = time.time()
    print('----------------start recommending...----------------')
    for episode in tqdm(range(args_r.test_epoches), total=args_r.test_epoches):
        current_states = env._initialize()

        model.reset(args_r.sigma)

        train_step = 0
        accumulate_loss = [0, 0]

        for st in tqdm(range(args_r.max_steps), total=args_r.max_steps):
            states = current_states

            actions = model.choose_action(states, True)

            rewards, states_, dones, _, _, _, _ = env._step(actions, final_data_feature, best_performance_file, best_paras_file)

            next_states = states_

            model.add_sample(states, actions, rewards, next_states, dones)

            current_states = next_states

            if len(model.replay_memory) > args_r.batch_size:
                losses = []
                for i in range(2):
                    loss = model.update()
                    if (model.update_time % model.delay_time) == 0:
                        losses.append(loss)
                        train_step += 1

                accumulate_loss[0] += sum([x[0] for x in losses])
                accumulate_loss[1] += sum([x[1] for x in losses])

            if env.nochange_steps == args_r.nochange_steps:
                break

        if env.steps == args_r.nochange_steps:
            env.nochange_episodes += 1
        else:
            env.nochange_episodes = 0

        if env.nochange_episodes == args_r.nochange_episodes:
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

    with open(episode_score_file, 'wb') as f:
        pickle.dump(episode_score, f)

    with open(episode_steps_file, 'wb') as f:
        pickle.dump(episode_steps, f)

    with open(episode_closs_file, 'wb') as f:
        pickle.dump(episode_closs, f)

    with open(episode_aloss_file, 'wb') as f:
        pickle.dump(episode_aloss, f)

    end_time = time.time()
    reccomend_time = end_time - start_time
    logger.info("recommend time: {}s".format(reccomend_time))





