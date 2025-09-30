'''
Query performance data collection for the QD scenario.
'''

import os
import shutil
import subprocess
import numpy as np
from tqdm import tqdm
import random

if __name__ == '__main__':
    s = 20
    np.random.seed(s)
    random.seed(s)

    compile_command = ['g++', '-Ofast', '-lrt', '-std=c++11', '-DHAVE_CXX0X', '-march=native', '-fpic', '-w',
                       '-fopenmp', '-ftree-vectorize', '-ftree-vectorizer-verbose=0', 'query_performance_collect_qd_change.cpp', '-o',
                       'query_performance_collect_qd_change'
                       ]
    subprocess.run(compile_command, check=True)
    print('Compilation completed.')

    # efCs = [20, 40, 60, 80, 100, 140, 180, 220, 260, 300, 340, 380, 420, 460, 500, 560, 620, 680, 740, 800]
    # ms = [4, 8, 12, 16, 20, 24, 32, 40, 48, 56, 64, 80, 100]
    efCs = [100]
    ms = [32]

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

    index_performance_csv = "./Data/index_performance.csv"
    if not os.path.exists(index_performance_csv):
        shutil.copy('./Data/index_performance_empty.csv', index_performance_csv)

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

                        if 1e5 <= size < 2e6 and dim in [96, 100, 128, 200, 300, 420, 960] and len(filename_list) == 3:
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
                                efC, m = para

                                tasks.append((whole_name, base_path, filename, query_path1, query_path2, query_path3, query_path4, indice_path1, indice_path2, indice_path3, indice_path4,
                                              index_performance_csv, subdir, size, dim, efC, m))

    for task in tqdm(tasks, total=len(tasks)):
        whole_name, base_path, filename, query_path1, query_path2, query_path3, query_path4, indice_path1, indice_path2, indice_path3, indice_path4, index_performance_csv, subdir, size, dim, efC, m = task

        index_path = os.path.join('./Index', '{}/{}_{}_{}.bin'.format(subdir, filename, efC, m))

        run_command = ['./query_performance_collect_qd_change', whole_name, base_path, query_path1, query_path2, query_path3, query_path4, indice_path1, indice_path2, indice_path3, indice_path4, index_path, index_performance_csv, subdir, str(size), str(dim), str(efC), str(m)]

        print(f'{whole_name}_{efC}_{m}')
        print('-------------------Start constructing the index and executing queries...-------------------')
        result = subprocess.run(run_command, check=True, text=True, capture_output=True)
        print('-------------------Index construction and querying completed-------------------')













