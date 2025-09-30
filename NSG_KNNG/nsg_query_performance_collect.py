import os
import shutil
import subprocess
import pandas as pd
import struct
import numpy as np
from tqdm import tqdm
import time
import random

'''
Collect query performance data corresponding to candidate construction parameter configurations on the given datasets for NSG index.
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

if __name__ == '__main__':
    s = 20
    np.random.seed(s)
    random.seed(s)

    # parameters of KNNG
    Ks = [100, 200, 300, 400]
    Ls = [100, 150, 200, 250, 300, 350, 400]

    ite = 12
    S = 15
    R = 100

    # construction parameters of NSG
    L_nsg_Cs = [150, 200, 250, 300, 350]
    R_nsgs = [5, 10, 15, 20, 25, 30, 35, 40, 50, 70, 90]
    Cs = [300, 400, 500, 600]

    # search parameters of NSG
    L_nsg_Ss = [10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320, 340, 360, 380, 400,
                430, 460, 490, 520, 550, 580, 610, 640, 670, 700, 740, 780, 820, 860, 900, 960, 1020, 1080, 1140, 1200, 1300, 1400, 1500]

    KNN_para_list = []
    NSG_para_list = []
    for L in Ls:
        for K in Ks:
            if K <= L:
                KNN_para_list.append((K, L))
            else:
                break

    for L_nsg_C in L_nsg_Cs:
        for R_nsg in R_nsgs:
            for C in Cs:
                if R_nsg <= C:
                    NSG_para_list.append((L_nsg_C, R_nsg, C))
                else:
                    break

    base_dir = "../Data/Base"
    query_dir = "../Data/Query"
    groundtruth_dir = "../Data/GroundTruth"
    KNN_graph_dir = "./KNN_graph"
    NSG_graph_dir = "./NSG_graph"

    os.makedirs(KNN_graph_dir, exist_ok=True)
    os.makedirs(NSG_graph_dir, exist_ok=True)

    '''
    index_performance_csv1:
        Record the construction time, storage usage, and other information for constructing KNNG on each dataset under different construction parameter configurations.
    index_performance_csv2:
        Record the construction time, storage usage, and other information for constructing NSG on each dataset under a fixed KNNG and different construction parameter configurations.
    index_performance_csv3:
        Record the search time, storage usage, query time and other information for each dataset under a fixed NSG and different search parameters.
    index_performance_csv4:
        Record the query recall for each dataset under a fixed NSG and different search parameters.
    '''
    index_performance_csv1 = "./Data/index_performance_KNNG.csv"
    index_performance_csv2 = "./Data/index_performance_NSG.csv"
    index_performance_csv3= "./Data/index_performance_Search.csv"
    index_performance_csv4= "./Data/index_performance_Search_Recall.csv"

    if not os.path.exists(index_performance_csv1):
        shutil.copy('./Data/index_performance_KNNG_empty.csv', index_performance_csv1)
    if not os.path.exists(index_performance_csv2):
        shutil.copy('./Data/index_performance_NSG_empty.csv', index_performance_csv2)
    if not os.path.exists(index_performance_csv3):
        shutil.copy('./Data/index_performance_Search_empty.csv', index_performance_csv3)
    if not os.path.exists(index_performance_csv4):
        shutil.copy('./Data/index_performance_Search_Recall_empty.csv', index_performance_csv4)

    whole_index_performance_csv = './Data/index_performance.csv'

    df = pd.read_csv(index_performance_csv2, sep=',', header=0)
    exist_para = df[['FileName', 'K', 'L', 'L_nsg_C', 'R_nsg', 'C']].to_numpy().tolist()

    for subdir in ['gist']:
        os.makedirs(os.path.join(KNN_graph_dir, subdir), exist_ok=True)
        os.makedirs(os.path.join(NSG_graph_dir, subdir), exist_ok=True)

        subdir_path = os.path.join(base_dir, subdir)
        if os.path.isdir(subdir_path):
            if os.listdir(subdir_path):
                for file_text in os.listdir(subdir_path):
                    filename = os.path.splitext(file_text)[0]

                    whole_name = subdir + '_' + filename

                    filename_list = filename.split('_')
                    if len(filename_list) == 2:
                        print(os.path.join(subdir_path, file_text))
                    else:
                        level = int(filename_list[0])
                        num = float(filename_list[1])
                        dim = int(filename_list[2])

                        size = int(pow(10, level) * 100000 * num)

                        if 1e5 <= size < 3e5:
                            base_path = os.path.join(subdir_path, file_text)
                            query_path = os.path.join(query_dir, '{}/{}.fvecs'.format(subdir, dim))
                            gt_indice_path = os.path.join(groundtruth_dir, '{}/{}.ivecs'.format(subdir, filename))

                            for KNN_para in tqdm(KNN_para_list, total = len(KNN_para_list)):
                                K, L = KNN_para
                                KNN_graph_path = os.path.join(KNN_graph_dir, '{}/{}_{}_{}.graph'.format(subdir, filename, K, L))
                                KNN_graph_time = 0

                                for NSG_para in tqdm(NSG_para_list, total = len(NSG_para_list)):
                                    L_nsg_C, R_nsg, C = NSG_para

                                    if [whole_name, K, L, L_nsg_C, R_nsg, C] not in exist_para:
                                        NSG_graph_path = os.path.join(NSG_graph_dir, '{}/{}_{}_{}_{}.nsg'.format(subdir, filename, L_nsg_C, R_nsg, C))

                                        print(f'{whole_name}_{K}_{L}_{L_nsg_C}_{R_nsg}_{C}')

                                        if not os.path.exists(KNN_graph_path):
                                            print(f'-------------------start constructing KNNG: {K}_{L}，only once-------------------')
                                            t1 = time.time()
                                            run_command1 = ['./FFANNA_KNNG/build/tests/test_nndescent', base_path, KNN_graph_path, str(K), str(L), str(ite), str(S), str(R), whole_name, index_performance_csv1]
                                            result1 = subprocess.run(run_command1, check=True, text=True, capture_output=True)
                                            t2 = time.time()

                                            KNN_graph_time = t2 - t1

                                        print(f'-------------------start constructing NSG: {L_nsg_C}_{R_nsg}_{C}-------------------')
                                        NSG_graph_time = 0
                                        if not os.path.exists(NSG_graph_path):
                                            t3 = time.time()
                                            run_command2 = ['./NSG/build/tests/test_nsg_index', base_path, KNN_graph_path, str(L_nsg_C), str(R_nsg), str(C), NSG_graph_path, str(K), str(L), whole_name, index_performance_csv2]
                                            result2 = subprocess.run(run_command2, check=True, text=True, capture_output=True)
                                            t4 = time.time()

                                            NSG_graph_time = t4 - t3

                                        print('-------------------start searching-------------------')
                                        recs = []
                                        for L_nsg_S in L_nsg_Ss:
                                            qs_indice_path = os.path.join(NSG_graph_dir, '{}/{}_{}_{}_{}_{}.nsg'.format(subdir, filename, L_nsg_C, R_nsg, C, L_nsg_S))

                                            run_command3 = ['./NSG/build/tests/test_nsg_optimized_search', base_path, query_path, NSG_graph_path, str(L_nsg_S), str(10), qs_indice_path, str(K), str(L), str(L_nsg_C), str(R_nsg), str(C), whole_name, index_performance_csv3]
                                            result3 = subprocess.run(run_command3, check=True, text=True, capture_output=True)

                                            gt = read_ivecs(gt_indice_path)
                                            qs = read_ivecs(qs_indice_path)

                                            rec = calculate_recall_rate(gt, qs)
                                            recs.append((whole_name, K, L, L_nsg_C, R_nsg, C, L_nsg_S, rec))

                                            if L_nsg_S >= 300 and rec >= 0.995:
                                                break

                                        rec_df = pd.DataFrame(recs, columns=['FileName', 'K', 'L', 'L_nsg_C', 'R_nsg', 'C', 'L_nsg_S', 'recall'])
                                        rec_df.to_csv(index_performance_csv4, mode='a', header=False, index=False)

                                        if NSG_graph_time < 1800:
                                            os.remove(NSG_graph_path)

    df_KNNG = pd.read_csv(index_performance_csv1, sep=',', header=0)
    df_NSG = pd.read_csv(index_performance_csv2, sep=',', header=0)
    df_Search = pd.read_csv(index_performance_csv3, sep=',', header=0)
    df_Search_Recall = pd.read_csv(index_performance_csv4, sep=',', header=0)

    df_merged = pd.merge(df_KNNG, df_NSG, on=['FileName', 'K', 'L'], how='left')
    df_merged = pd.merge(df_merged, df_Search, on=['FileName', 'K', 'L', 'L_nsg_C', 'R_nsg', 'C'], how='left')
    df_merged = pd.merge(df_merged, df_Search_Recall, on=['FileName', 'K', 'L', 'L_nsg_C', 'R_nsg', 'C', 'L_nsg_S'], how='left')
    df_merged.to_csv(whole_index_performance_csv, mode='w', header=True, index=False)












