'''
Collect query performance data corresponding to candidate construction parameter configurations on the given datasets.
'''

import os
import shutil
import subprocess
import pandas as pd
import numpy as np
from tqdm import tqdm
import random

if __name__ == '__main__':
    s = 20
    np.random.seed(s)
    random.seed(s)

    compile_command = ['g++', '-Ofast', '-lrt', '-std=c++11', '-DHAVE_CXX0X', '-march=native', '-fpic', '-w',
                       '-fopenmp', '-ftree-vectorize', '-ftree-vectorizer-verbose=0', 'query_performance_collect.cpp', '-o',
                       'query_performance_collect'
                       ]
    subprocess.run(compile_command, check=True)
    print('Compilation completed.')

    efCs = [20, 40, 60, 80, 100, 140, 180, 220, 260, 300, 340, 380, 420, 460, 500, 560, 620, 680, 740, 800]
    ms = [4, 8, 12, 16, 20, 24, 32, 40, 48, 56, 64, 80, 100]

    # For the GridSearch method, we use ms = [4, 8, 16, 24, 32, 48, 64, 80, 100] on the SIFT50M dataset
    # in our experiments to accelerate data collection.

    para_list = []
    for efC in efCs:
        for m in ms:
            if m <= efC:
                para_list.append((efC, m))
            else:
                break

    tasks = []

    base_dir = "./Data/Base"
    query_dir = "./Data/Query"
    groundtruth_dir = "./Data/GroundTruth"
    index_dir = "./Index"

    os.makedirs(index_dir, exist_ok=True)

    '''
    index_performance_csv:
    This file stores the data of the constructed indexes across all construction parameter configurations on each dataset,
    including basic data, index construction time, storage usage, query recall, query time, and other metrics.
    
    '''
    index_performance_csv = "./Data/index_performance.csv"

    if not os.path.exists(index_performance_csv):
        shutil.copy('./Data/index_performance_empty.csv', index_performance_csv)

    df = pd.read_csv(index_performance_csv, sep=',', header=0)
    exist_para = df[['FileName', 'efConstruction', 'M']].to_numpy().tolist()

    # Please replace the dataset names in the list with the datasets from which you need to collect data.
    for subdir in ['gist']:
        os.makedirs(os.path.join(index_dir, subdir), exist_ok=True)

        subdir_path = os.path.join(base_dir, subdir)
        if os.path.isdir(subdir_path):
            if os.listdir(subdir_path):
                for file_text in os.listdir(subdir_path):
                    filename = os.path.splitext(file_text)[0]

                    whole_name = subdir + '_' + filename
                    filename_list = filename.split('_')

                    if len(filename_list) == 2:
                        pass
                    else:
                        level = int(filename_list[0])
                        num = float(filename_list[1])
                        dim = int(filename_list[2])
                        size = int(pow(10, level) * 100000 * num)

                        if 1e6 <= size < 2e6 and dim in [96, 100, 128, 200, 256, 300, 384, 420, 960] and len(filename_list) == 3:
                            base_path = os.path.join(subdir_path, file_text)
                            indice_path = os.path.join(groundtruth_dir, '{}/{}.ivecs'.format(subdir, filename))

                            if subdir in ['deep', 'glove', 'sift', 'paper', 'nytimes', 'crawl', 'tiny', 'msong', 'gist']:
                                if dim == 128:
                                    query_path = os.path.join(query_dir, '{}/{}.bvecs'.format(subdir, dim))
                                else:
                                    query_path = os.path.join(query_dir, '{}/{}.fvecs'.format(subdir, dim))
                            else:
                                query_path = os.path.join(query_dir, '{}/{}.fvecs'.format(subdir, filename))

                            for para in para_list:
                                efC, m = para
                                if [whole_name, efC, m] not in exist_para:
                                    tasks.append((whole_name, base_path, query_path, indice_path, filename, index_performance_csv,
                                                  subdir, size, dim, efC, m))

    for task in tqdm(tasks, total=len(tasks)):
        whole_name, base_path, query_path, indice_path, filename, index_performance_csv, subdir, size, dim, efC, m = task
        index_path = os.path.join('./Index', '{}/{}_{}_{}.bin'.format(subdir, filename, efC, m))

        run_command = ['./query_performance_collect', whole_name, base_path, query_path, indice_path, index_path, index_performance_csv,
                       subdir, str(size), str(dim), str(efC), str(m)]

        print(f'{whole_name}_{efC}_{m}')
        print('-------------------Start constructing the index and executing queries...-------------------')
        result = subprocess.run(run_command, check=True, text=True, capture_output=True)
        print('-------------------Index construction and querying completed-------------------')













