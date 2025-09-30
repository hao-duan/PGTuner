'''
Construct the index based on the construction parameter configuration in the recommended configuration and execute queries for the QD scenario.
'''

import os
import shutil
import subprocess
import pandas as pd
import numpy as np
from tqdm import tqdm
import random
import re
from query_performance_predict.Args import args as args_p
from parameter_configuration_recommend.Args import args as args_r


def get_recommended_configurations():
    store_dir = './parameter_configuration_recommend/{}_{}_TD3'.format(args_r.actor_layer_sizes, args_r.critic_layer_sizes)

    dataset_name = args_r.dataset_name
    experiment_mode = args_p.experiment_mode
    store_subdir = os.path.join(store_dir, '{}'.format(experiment_mode))

    expr_name = '{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(args_r.epoches, args_r.max_steps, args_r.batch_size, args_r.alr,
                                                    args_r.tau, args_r.sigma, args_r.delay_time, args_r.pec_reward, args_r.nochange_steps)

    result_path = os.path.join(store_subdir, 'recommend_results/eval_{}_{}_{}_{}.csv'.format(
        expr_name, dataset_name, args_r.test_epoches, args_r.nochange_episodes, args_r.test_epoches))

    df = pd.read_csv(result_path)

    config_dic = {}
    config_df = df[['FileName', 'target_recall', 'efConstruction', 'M', 'efSearch']]

    groups = config_df.groupby('FileName')
    for filename, group in groups:
        config_dic[filename] = group[['target_recall', 'efConstruction', 'M', 'efSearch']].values.tolist()

    print(f'The recommended configurations for {dataset_name}:')
    print(config_dic)

    return config_dic

if __name__ == '__main__':
    s = 20
    np.random.seed(s)
    random.seed(s)

    compile_command = ['g++', '-Ofast', '-lrt', '-std=c++11', '-DHAVE_CXX0X', '-march=native', '-fpic', '-w',
                       '-fopenmp', '-ftree-vectorize', '-ftree-vectorizer-verbose=0', 'query_performance_verify_qd_change.cpp', '-o',
                       'query_performance_verify_qd_change'
                       ]
    subprocess.run(compile_command, check=True)
    print('Compilation completed.')

    filename_dic = {'deep1':'1_1_96_1', 'sift1':'1_1_128_1', 'glove': '1_1.183514_100', 'paper':'1_2.029997_200', 'crawl':'1_1.989995_300', 'msong':'-1_1_420_1', 'nytimes':'0_2.9_256',
                    'tiny':'1_1_384', 'gist':'1_1.0_960', 'deep10':'2_1_96', 'sift50':'2_5_128_1', 'sift2':'1_2_128_1', 'sift3':'1_3_128_1', 'sift4':'1_4_128_1', 'sift5':'1_5_128_1',
                    'gist_25':'1_1.0_960_25', 'gist_50':'1_1.0_960_50', 'gist_75':'1_1.0_960_75', 'gist_100':'1_1.0_960_100'}

    base_dir = "./Data/Base"
    query_dir = "./Data/Query"
    groundtruth_dir = "./Data/GroundTruth"
    index_dir = "./Index"

    os.makedirs(index_dir, exist_ok=True)

    experiment_mode = args_p.experiment_mode
    os.makedirs('./Data/experiments_results', exist_ok=True)
    os.makedirs('./Data/experiments_results/{}'.format(experiment_mode), exist_ok=True)

    index_performance_csv = "./Data/experiments_results/{}/index_performance_verify.csv".format(experiment_mode)
    if not os.path.exists(index_performance_csv):
        shutil.copy('./Data/index_performance_verify_empty.csv', index_performance_csv)

    tasks = []

    config_dic = get_recommended_configurations()

    for dataset_name in config_dic.keys():
        para_list = config_dic[dataset_name]

        filename = filename_dic[dataset_name]

        subdir = re.match(r'\D+', dataset_name).group()

        os.makedirs(os.path.join(index_dir, subdir), exist_ok=True)

        subdir_path = os.path.join(base_dir, subdir)

        if os.path.isdir(subdir_path):
            if subdir == 'sift':
                file_text = filename + '.bvecs'
            else:
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

                    filename1 = filename + '_25'
                    filename2 = filename + '_50'
                    filename3 = filename + '_75'
                    filename4 = filename + '_100'

                    indice_path1 = os.path.join(groundtruth_dir, '{}/{}.ivecs'.format(subdir, filename1))
                    indice_path2 = os.path.join(groundtruth_dir, '{}/{}.ivecs'.format(subdir, filename2))
                    indice_path3 = os.path.join(groundtruth_dir, '{}/{}.ivecs'.format(subdir, filename3))
                    indice_path4 = os.path.join(groundtruth_dir, '{}/{}.ivecs'.format(subdir, filename4))

                    if subdir in ['deep', 'glove', 'sift', 'paper', 'nytimes', 'crawl', 'tiny', 'msong', 'gist']:
                        if dim == 128:
                            query_path1 = os.path.join(query_dir, '{}/{}_25.bvecs'.format(subdir, dim))
                            query_path2 = os.path.join(query_dir, '{}/{}_50.bvecs'.format(subdir, dim))
                            query_path3 = os.path.join(query_dir, '{}/{}_75.bvecs'.format(subdir, dim))
                            query_path4 = os.path.join(query_dir, '{}/{}_100.bvecs'.format(subdir, dim))
                        else:
                            query_path1 = os.path.join(query_dir, '{}/{}_25.fvecs'.format(subdir, dim))
                            query_path2 = os.path.join(query_dir, '{}/{}_50.fvecs'.format(subdir, dim))
                            query_path3 = os.path.join(query_dir, '{}/{}_75.fvecs'.format(subdir, dim))
                            query_path4 = os.path.join(query_dir, '{}/{}_100.fvecs'.format(subdir, dim))

                    for para in para_list:
                        target_recall, efC, m, efS = para
                        efC = int(efC)
                        m = int(m)
                        efS = int(efS)

                        tasks.append((whole_name, base_path, filename, query_path1, query_path2, query_path3, query_path4, indice_path1, indice_path2, indice_path3, indice_path4,
                                          index_performance_csv, subdir, size, dim, efC, m, efS, target_recall))

    for task in tqdm(tasks, total=len(tasks)):
        whole_name, base_path, filename, query_path1, query_path2, query_path3, query_path4, indice_path1, indice_path2, indice_path3, indice_path4, index_performance_csv, subdir, size, dim, efC, m, efS, target_recall = task

        index_path = os.path.join('./Index', '{}/{}_{}_{}.bin'.format(subdir, filename, efC, m))

        run_command = ['./query_performance_verify_qd_change', whole_name, base_path, query_path1, query_path2, query_path3, query_path4, indice_path1, indice_path2, indice_path3, indice_path4,
                       index_path, index_performance_csv, subdir, str(size), str(dim), str(efC), str(m), str(efS), str(target_recall)]

        print(f'{whole_name}_{target_recall}_{efC}_{m}_{efS}')
        print('-------------------Start constructing the index and executing queries...-------------------')
        result = subprocess.run(run_command, check=True, text=True, capture_output=True)
        print('-------------------Index construction and querying completed-------------------')













