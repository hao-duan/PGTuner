import sys
sys.path.append('../')

import os
import pickle
import numpy as np
import pandas as pd
from Args import args as args_r
from query_performance_predict.Args import args as args_p

def get_pr_results(best_performance_file, best_paras_file, result_path, dataset_name, target_rec_lis):  # flag为0表示不是serial，为1表示是
    columns = ['efConstruction', 'M', 'efSearch', 'recall', 'average_ct_dc_counts', 'average_st_dc_counts']

    with open(best_paras_file, 'rb') as f:
        best_paras = pickle.load(f)

    with open(best_performance_file, 'rb') as f:
        best_performance = pickle.load(f)

    result = np.concatenate((best_paras, best_performance), axis=1)

    df = pd.DataFrame(result, columns=columns)

    df['FileName'] = [dataset_name] * len(target_rec_lis)
    df['target_recall'] = target_rec_lis

    df.to_csv(result_path, mode='w', index=False)

    config_dic = {}
    config_df = df[['FileName', 'target_recall', 'efConstruction', 'M', 'efSearch']]

    groups = config_df.groupby('FileName')
    for filename, group in groups:
        config_dic[filename] = group[['target_recall', 'efConstruction', 'M', 'efSearch']].values.tolist()

    return config_dic

def get_pr_results_nsg(best_performance_file, best_paras_file, result_path, dataset_name, target_rec_lis):
    columns = ['K', 'L', 'L_nsg_C', 'R_nsg', 'C', 'L_nsg_S', 'recall', 'average_NSG_s_dc_counts']

    with open(best_paras_file, 'rb') as f:
        best_paras = pickle.load(f)

    with open(best_performance_file, 'rb') as f:
        best_performance = pickle.load(f)

    result = np.concatenate((best_paras, best_performance), axis=1)

    df = pd.DataFrame(result, columns=columns)

    df['FileName'] = [dataset_name] * len(target_rec_lis)
    df['target_recall'] = target_rec_lis

    df.to_csv(result_path, mode='w', index=False)

    config_dic = {}
    config_df = df[['FileName', 'target_recall', 'K', 'L', 'L_nsg_C', 'R_nsg', 'C', 'L_nsg_S']]

    groups = config_df.groupby('FileName')
    for filename, group in groups:
        config_dic[filename] = group[['target_recall', 'K', 'L', 'L_nsg_C', 'R_nsg', 'C', 'L_nsg_S']].values.tolist()

    return config_dic

if __name__ == '__main__':
    current_directory = os.getcwd()
    parent_directory = os.path.dirname(current_directory)

    target_rec_lis = [0.85, 0.88, 0.9, 0.92, 0.94, 0.95, 0.96, 0.98, 0.99]
    # target_rec_lis = [0.9, 0.95, 0.99]
    dataset_name = args_r.dataset_name

    store_dir = './{}_{}_TD3'.format(args_r.actor_layer_sizes, args_r.critic_layer_sizes)
    # store_dir = './{}_{}_TD3_nsg'.format(args_r.actor_layer_sizes, args_r.critic_layer_sizes)

    experiment_mode = args_p.experiment_mode

    store_subdir = os.path.join(store_dir, '{}'.format(experiment_mode))
    os.makedirs(store_subdir, exist_ok=True)

    os.makedirs(os.path.join(store_subdir, 'recommend_results'), exist_ok=True)

    expr_name = '{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(args_r.epoches, args_r.max_steps, args_r.batch_size, args_r.alr,
                                                    args_r.tau, args_r.sigma, args_r.delay_time, args_r.pec_reward, args_r.nochange_steps)

    best_performance_file = os.path.join(store_subdir,
                                         'eval_best_performance_{}_{}_{}_{}.pkl'.format(expr_name, dataset_name, args_r.nochange_episodes, args_r.test_epoches))
    best_paras_file = os.path.join(store_subdir,
                                   'eval_best_paras_{}_{}_{}_{}.pkl'.format(expr_name, dataset_name, args_r.nochange_episodes, args_r.test_epoches))

    result_path = os.path.join(store_subdir ,
                               'recommend_results/eval_{}_{}_{}_{}.csv'.format(expr_name, dataset_name, args_r.test_epoches,
                                                                         args_r.nochange_episodes, args_r.test_epoches))

    config_dic= get_pr_results(best_performance_file, best_paras_file, result_path, dataset_name, target_rec_lis)
    # config_dic= get_pr_results_nsg(best_performance_file, best_paras_file, result_path, dataset_name, target_rec_lis)

    print(f'The recommended configurations for {dataset_name}:')
    print(config_dic)