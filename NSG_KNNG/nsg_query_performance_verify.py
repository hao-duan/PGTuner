import sys
sys.path.append('../')

import os
import shutil
import subprocess
import pandas as pd
import struct
import numpy as np
from tqdm import tqdm
import time
import random
import re
from query_performance_predict.Args import args as args_p
from parameter_configuration_recommend.Args import args as args_r

'''
Construct the index based on the construction parameter configuration in the recommended configuration and execute queries. 
Search for the minimum L that can achieve the target recall on the basis of the recommended L, 
and record data including basic data, index construction time, storage usage, target recall, query recall, query time, and other metrics.
'''

def read_ivecs(file_path):
    indices = []
    with open(file_path, 'rb') as f:
        while True:
            k_bytes = f.read(4)
            if not k_bytes:
                break
            k, = struct.unpack('I', k_bytes)
            vector_bytes = f.read(k * 4)
            indice = np.frombuffer(vector_bytes, dtype=np.int32)
            indices.append(indice)
    return np.array(indices)

def calculate_recall_rate(gt, qs):
    recall_rates = []

    K = qs.shape[1]
    gt = gt[:, :K]

    for row1, row2 in zip(gt, qs):
        recall_count = sum(elem in row1 for elem in row2)
        recall_rate = recall_count / K
        recall_rates.append(recall_rate)

    average_recall = np.mean(recall_rates)
    return average_recall

def get_recommended_configurations():
    store_dir = '../parameter_configuration_recommend/{}_{}_TD3_nsg'.format(args_r.actor_layer_sizes, args_r.critic_layer_sizes)

    dataset_name = args_r.dataset_name
    experiment_mode = args_p.experiment_mode
    store_subdir = os.path.join(store_dir, '{}'.format(experiment_mode))

    expr_name = '{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(args_r.epoches, args_r.max_steps, args_r.batch_size, args_r.alr,
                                                    args_r.tau, args_r.sigma, args_r.delay_time, args_r.pec_reward,
                                                    args_r.nochange_steps)

    result_path = os.path.join(store_subdir, 'recommend_results/eval_{}_{}_{}_{}.csv'.format(
                                       expr_name, dataset_name, args_r.test_epoches, args_r.nochange_episodes, args_r.test_epoches))

    df = pd.read_csv(result_path)

    config_dic = {}
    config_df = df[['FileName', 'target_recall', 'K', 'L', 'L_nsg_C', 'R_nsg', 'C', 'L_nsg_S']]

    groups = config_df.groupby('FileName')
    for filename, group in groups:
        config_dic[filename] = group[['target_recall', 'K', 'L', 'L_nsg_C', 'R_nsg', 'C', 'L_nsg_S']].values.tolist()

    print(f'The recommended configurations for {dataset_name}:')
    print(config_dic)

    return config_dic

if __name__ == '__main__':
    s = 20
    np.random.seed(s)
    random.seed(s)

    filename_dic = {'gist1':'1_1.0_960', 'gist':'0_1_960'}

    ite = 12
    S = 15
    R = 100

    base_dir = "../Data/Base"
    query_dir = "../Data/Query"
    groundtruth_dir = "../Data/GroundTruth"
    KNN_graph_dir = "./KNN_graph"
    NSG_graph_dir = "./NSG_graph"

    os.makedirs(KNN_graph_dir, exist_ok=True)
    os.makedirs(NSG_graph_dir, exist_ok=True)

    experiment_mode = args_p.experiment_mode
    os.makedirs('./Data/experiments_results', exist_ok=True)
    os.makedirs('./Data/experiments_results/{}'.format(experiment_mode), exist_ok=True)

    index_performance_csv1 = "./Data/experiments_results/{}/index_performance_KNNG_verify.csv".format(experiment_mode)
    index_performance_csv2 = "./Data/experiments_results/{}/index_performance_NSG_verify.csv".format(experiment_mode)
    index_performance_csv3= "./Data/experiments_results/{}/index_performance_Search_verify.csv".format(experiment_mode)
    index_performance_csv4= "./Data/experiments_results/{}/index_performance_Search_Recall_verify.csv".format(experiment_mode)

    if not os.path.exists(index_performance_csv1):
        shutil.copy('./Data/index_performance_KNNG_verify_empty.csv', index_performance_csv1)
    if not os.path.exists(index_performance_csv2):
        shutil.copy('./Data/index_performance_NSG_verify_empty.csv', index_performance_csv2)
    if not os.path.exists(index_performance_csv3):
        shutil.copy('./Data/index_performance_Search_verify_empty.csv', index_performance_csv3)
    if not os.path.exists(index_performance_csv4):
        shutil.copy('./Data/index_performance_Search_Recall_verify_empty.csv', index_performance_csv4)

    whole_index_performance_csv = './Data/experiments_results/{}/index_performance_verify.csv'.format(experiment_mode)

    df = pd.read_csv(index_performance_csv2, sep=',', header=0)
    exist_para = df[['FileName', 'K', 'L', 'L_nsg_C', 'R_nsg', 'C']].to_numpy().tolist()

    config_dic = get_recommended_configurations()
    for dataset_name in config_dic.keys():
        para_list = config_dic[dataset_name]

        filename = filename_dic[dataset_name]
        subdir = re.match(r'\D+', dataset_name).group()

        os.makedirs(os.path.join(KNN_graph_dir, subdir), exist_ok=True)
        os.makedirs(os.path.join(NSG_graph_dir, subdir), exist_ok=True)

        subdir_path = os.path.join(base_dir, subdir)
        if os.path.isdir(subdir_path):
            file_text = filename + '.fvecs'

            whole_name = subdir + '_' + filename

            filename_list = filename.split('_')
            if len(filename_list) == 2:
                print(os.path.join(subdir_path, file_text))
            else:
                level = int(filename_list[0])
                num = float(filename_list[1])
                dim = int(filename_list[2])

                size = int(pow(10, level) * 100000 * num)

                if dim in [96, 100, 128, 200, 256, 300, 384, 420, 960]:
                    base_path = os.path.join(subdir_path, file_text)
                    gt_indice_path = os.path.join(groundtruth_dir, '{}/{}.ivecs'.format(subdir, filename))

                    query_path = os.path.join(query_dir, '{}/{}.fvecs'.format(subdir, dim))

                    for para in tqdm(para_list, total = len(para_list)):
                        target_recall, K, L, L_nsg_C, R_nsg, C, pr_L_nsg_S = para
                        K = int(K)
                        L = int(L)
                        L_nsg_C = int(L_nsg_C)
                        R_nsg = int(R_nsg)
                        C = int(C)
                        pr_L_nsg_S = int(pr_L_nsg_S)

                        if [whole_name, target_recall, K, L, L_nsg_C, R_nsg, C, pr_L_nsg_S] not in exist_para:
                            KNN_graph_path = os.path.join(KNN_graph_dir, '{}/{}_{}_{}.graph'.format(subdir, filename, K, L))
                            NSG_graph_path = os.path.join(NSG_graph_dir, '{}/{}_{}_{}_{}.nsg'.format(subdir, filename, L_nsg_C, R_nsg, C))

                            print(f'-------------------start constructing KNNG: {K}_{L}，only once-------------------')
                            t1 = time.time()
                            run_command1 = ['./FFANNA_KNNG/build/tests/test_nndescent', base_path, KNN_graph_path, str(K), str(L), str(ite), str(S), str(R), whole_name, index_performance_csv1]
                            result1 = subprocess.run(run_command1, check=True, text=True, capture_output=True)
                            t2 = time.time()

                            KNN_graph_time = t2 - t1

                            print(f'-------------------start constructing NSG: {L_nsg_C}_{R_nsg}_{C}-------------------')
                            t3 = time.time()
                            run_command2 = ['./NSG/build/tests/test_nsg_index', base_path, KNN_graph_path, str(L_nsg_C), str(R_nsg), str(C), NSG_graph_path, str(K), str(L), whole_name, index_performance_csv2]
                            result2 = subprocess.run(run_command2, check=True, text=True, capture_output=True)
                            t4 = time.time()

                            NSG_graph_time = t4 - t3

                            print('-------------------start searching-------------------')
                            dealt = 0
                            if (pr_L_nsg_S < 100):
                                dealt = 1
                            elif (100 <= pr_L_nsg_S < 200):
                                dealt = 5
                            elif (200 <= pr_L_nsg_S < 400):
                                dealt = 10
                            elif (400 <= pr_L_nsg_S < 700):
                                dealt = 15
                            elif (700 <= pr_L_nsg_S < 900):
                                dealt = 20
                            elif (900 <= pr_L_nsg_S < 1200):
                                dealt = 30
                            elif (1200 <= pr_L_nsg_S < 1500):
                                dealt = 50
                            else:
                                dealt = 100


                            qs_indice_path = os.path.join(NSG_graph_dir, '{}/{}_{}_{}_{}_{}.nsg'.format(subdir, filename, L_nsg_C, R_nsg, C, pr_L_nsg_S))

                            run_command3 = ['./NSG/build/tests/test_nsg_optimized_search', base_path, query_path, NSG_graph_path, str(pr_L_nsg_S), str(10), qs_indice_path, str(K), str(L), str(L_nsg_C), str(R_nsg), str(C), whole_name, index_performance_csv3]
                            result3 = subprocess.run(run_command3, check=True, text=True, capture_output=True)

                            gt = read_ivecs(gt_indice_path)
                            qs = read_ivecs(qs_indice_path)

                            rec = calculate_recall_rate(gt, qs)

                            os.remove(qs_indice_path)
                            t5 = time.time()
                            init_paras_search_time = t5 - t4

                            recs = [(whole_name, K, L, L_nsg_C, R_nsg, C, pr_L_nsg_S, pr_L_nsg_S, target_recall, rec, init_paras_search_time)]

                            L_nsg_S = pr_L_nsg_S
                            if rec > target_recall:
                                while(True):
                                    if (L_nsg_S < 100):
                                        dealt = 1
                                    elif (100 <= L_nsg_S < 200):
                                        dealt = 5
                                    elif (200 <= L_nsg_S < 400):
                                        dealt = 10
                                    elif (400 <= L_nsg_S < 700):
                                        dealt = 15
                                    elif (700 <= L_nsg_S < 900):
                                        dealt = 20
                                    elif (900 <= L_nsg_S < 1200):
                                        dealt = 30
                                    elif (1200 <= L_nsg_S < 1500):
                                        dealt = 50
                                    else:
                                        dealt = 100

                                    L_nsg_S = L_nsg_S - dealt

                                    qs_indice_path = os.path.join(NSG_graph_dir, '{}/{}_{}_{}_{}_{}.nsg'.format(subdir, filename, L_nsg_C, R_nsg, C, pr_L_nsg_S))

                                    run_command3 = ['./NSG/build/tests/test_nsg_optimized_search', base_path,  query_path, NSG_graph_path, str(L_nsg_S), str(10),
                                                    qs_indice_path, str(K), str(L), str(L_nsg_C), str(R_nsg), str(C), whole_name, index_performance_csv3]
                                    result3 = subprocess.run(run_command3, check=True, text=True, capture_output=True)

                                    gt = read_ivecs(gt_indice_path)
                                    qs = read_ivecs(qs_indice_path)

                                    rec = calculate_recall_rate(gt, qs)

                                    os.remove(qs_indice_path)

                                    if rec >= target_recall:
                                        t6 = time.time()
                                        paras_search_time = t6 - t4

                                        df =  pd.read_csv(index_performance_csv3, sep=',', header=0)
                                        df = df.drop(df.index[-2])
                                        df.to_csv(index_performance_csv3, mode='w', index=False)

                                        recs = [(whole_name, K, L, L_nsg_C, R_nsg, C, pr_L_nsg_S, L_nsg_S, target_recall, rec, paras_search_time)]
                                    else:
                                        df = pd.read_csv(index_performance_csv3, sep=',', header=0)
                                        df = df.drop(df.index[-1])
                                        df.to_csv(index_performance_csv3, mode='w', index=False)

                                        break
                            else:
                                while (True):
                                    if (L_nsg_S < 100):
                                        dealt = 1
                                    elif (100 <= L_nsg_S < 200):
                                        dealt = 5
                                    elif (200 <= L_nsg_S < 400):
                                        dealt = 10
                                    elif (400 <= L_nsg_S < 700):
                                        dealt = 15
                                    elif (700 <= L_nsg_S < 900):
                                        dealt = 20
                                    elif (900 <= L_nsg_S < 1200):
                                        dealt = 30
                                    elif (1200 <= L_nsg_S < 1500):
                                        dealt = 50
                                    else:
                                        dealt = 100

                                    L_nsg_S = L_nsg_S + dealt

                                    qs_indice_path = os.path.join(NSG_graph_dir, '{}/{}_{}_{}_{}_{}.nsg'.format(subdir, filename, L_nsg_C, R_nsg, C, pr_L_nsg_S))

                                    run_command3 = ['./NSG/build/tests/test_nsg_optimized_search', base_path, query_path, NSG_graph_path, str(L_nsg_S), str(10),
                                                    qs_indice_path, str(K), str(L), str(L_nsg_C), str(R_nsg), str(C), whole_name, index_performance_csv3]
                                    result3 = subprocess.run(run_command3, check=True, text=True, capture_output=True)

                                    gt = read_ivecs(gt_indice_path)
                                    qs = read_ivecs(qs_indice_path)

                                    rec = calculate_recall_rate(gt, qs)

                                    os.remove(qs_indice_path)

                                    if rec >= target_recall:
                                        t6 = time.time()
                                        paras_search_time = t6 - t4

                                        df = pd.read_csv(index_performance_csv3, sep=',', header=0)
                                        df = df.drop(df.index[-2])
                                        df.to_csv(index_performance_csv3, mode='w', index=False)

                                        recs = [(whole_name, K, L, L_nsg_C, R_nsg, C, pr_L_nsg_S, L_nsg_S, target_recall, rec, paras_search_time)]

                                        break
                                    else:
                                        df = pd.read_csv(index_performance_csv3, sep=',', header=0)
                                        df = df.drop(df.index[-1])
                                        df.to_csv(index_performance_csv3, mode='w', index=False)

                            rec_df = pd.DataFrame(recs, columns=['FileName', 'K', 'L', 'L_nsg_C', 'R_nsg', 'C', 'pr_L_nsg_S', 'real_L_nsg_S', 'target_recall', 'real_recall', 'paras_search_time'])
                            rec_df.to_csv(index_performance_csv4, mode='a', header=False, index=False)

                            if NSG_graph_time < 1200:
                                os.remove(NSG_graph_path)

                            if KNN_graph_time < 1200:
                                os.remove(KNN_graph_path)

    df_KNNG = pd.read_csv(index_performance_csv1, sep=',', header=0)
    df_NSG = pd.read_csv(index_performance_csv2, sep=',', header=0)
    df_Search = pd.read_csv(index_performance_csv3, sep=',', header=0)
    df_Search_Recall = pd.read_csv(index_performance_csv4, sep=',', header=0)

    df_Search = df_Search.rename(columns={'L_nsg_S': 'real_L_nsg_S'})

    df_merged = pd.merge(df_KNNG, df_NSG, on=['FileName', 'K', 'L'], how='left')
    df_merged = pd.merge(df_merged, df_Search, on=['FileName', 'K', 'L', 'L_nsg_C', 'R_nsg', 'C'], how='left')
    df_merged = pd.merge(df_merged, df_Search_Recall, on=['FileName', 'K', 'L', 'L_nsg_C', 'R_nsg', 'C', 'real_L_nsg_S'], how='left')
    df_merged.to_csv(whole_index_performance_csv, mode='w', header=True, index=False)












