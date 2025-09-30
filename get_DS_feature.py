import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import random
import cupy as cp
from cuml.neighbors import NearestNeighbors
from utils.data_rw import read_bvecs, read_fvecs
from utils.data_sample import data_sample_sequential, data_sample_sequential_float
import math

def uniform_sample(max_num, sample_size):
    indices = np.random.choice(max_num, sample_size, replace=False)
    return indices

def get_k_neighbor_dists(vectors, k):
    vectors = np.unique(vectors, axis=0)
    vectors = cp.array(vectors)

    nn = NearestNeighbors(n_neighbors=k+1, algorithm='brute', metric='euclidean')
    nn.fit(vectors)

    distances, _ = nn.kneighbors(vectors)
    whole_mean_dist = distances[:, 1:-1].mean()
    distances = distances / whole_mean_dist

    sum_k_neighbor_dists = distances[:, 1:-1].sum(axis=1)

    sum_min_dist = sum_k_neighbor_dists.min()
    sum_max_dist = sum_k_neighbor_dists.max()
    sum_std_dist = sum_k_neighbor_dists.std()

    return sum_min_dist, sum_max_dist, sum_std_dist

def process_file(args, k):
    subdir, file_text, root_dir = args

    subdir_path = os.path.join(root_dir, subdir)
    filename = os.path.splitext(file_text)[0]
    whole_name = subdir + '_' + filename

    filename_list = filename.split('_')
    level = int(filename_list[0])
    num = float(filename_list[1])
    dim = int(filename_list[2])
    size = int(pow(10, level) * 100000 * num)

    file_path = os.path.join(subdir_path, file_text)

    if dim == 128 and subdir == 'sift':
        vectors = read_bvecs(file_path, None)
    else:
        vectors = read_fvecs(file_path, None)

    if vectors.dtype != np.float32:
        vectors = vectors.astype(np.float32)

    print('Start extracting the DS features...')
    if size <= 1e6:
        t1 = time.time()
        sum_min_dist, sum_max_dist, sum_std_dist = get_k_neighbor_dists(vectors, k)
        t2 = time.time()
        search_time = t2 - t1
    else:
        sampled_sum_min_dist_data = []
        sampled_sum_max_dist_data = []
        sampled_sum_std_dist_data = []

        if size < 1e7:
            sample_num = int(size / 1e6 + 0.5)
        else:
            sample_num = int(size / 5e6 + 0.5)
        sample_ids = uniform_sample(size, sample_num)
        t3 = time.time()
        for start_id in tqdm(sample_ids, total = len(sample_ids)):
            if size < 1e7:
                if subdir == 'sift':
                    sample_vectors = data_sample_sequential(file_path, dim, size, int(1e6), start_id)
                else:
                    sample_vectors = data_sample_sequential_float(file_path, dim, size, int(1e6), start_id)
            else:
                if subdir == 'sift':
                    sample_vectors = data_sample_sequential(file_path, dim, size, int(5e6), start_id)
                else:
                    sample_vectors = data_sample_sequential_float(file_path, dim, size, int(5e6), start_id)

            sample_sum_min_dist, sample_sum_max_dist, sample_sum_std_dist = get_k_neighbor_dists(sample_vectors, k)

            sampled_sum_min_dist_data.append(sample_sum_min_dist)
            sampled_sum_max_dist_data.append(sample_sum_max_dist)
            sampled_sum_std_dist_data.append(sample_sum_std_dist)

        sum_min_dist = sum(sampled_sum_min_dist_data) / sample_num
        sum_max_dist = sum(sampled_sum_max_dist_data) / sample_num
        sum_std_dist = sum(sampled_sum_std_dist_data) / sample_num

        t4 = time.time()
        search_time = t4 - t3

    feature_data = {
        "FileName": whole_name, "Sum_K_MinDist": sum_min_dist, "Sum_K_MaxDist": sum_max_dist, "Sum_K_StdDist": sum_std_dist, "SearchTime": search_time}
    return feature_data

if __name__ == '__main__':
    root_dir = "./Data/Base"
    K_neighbor_dist_feature_csv = "./Data/K_neighbor_dist_feature.csv"

    exist_name = []
    if os.path.exists(K_neighbor_dist_feature_csv):
        df = pd.read_csv(K_neighbor_dist_feature_csv, sep=',', header=0)
        exist_name = list(df['FileName'])

    file_tasks = []
    k = 100

    for subdir in os.listdir(root_dir):
        # Replace the dataset names in the list with the datasets from which you need to extract the DS features.
        if subdir in ['deep']:
            subdir_path = os.path.join(root_dir, subdir)
            if os.path.isdir(subdir_path) and os.listdir(subdir_path):
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

                        if 1e6 <= size < 2e6  and dim in [96, 100, 128, 200, 256, 300, 384, 420, 960] and whole_name not in exist_name and len(filename_list) == 3:
                            args = (subdir, file_text, root_dir)
                            file_tasks.append(args)

    for task in tqdm(file_tasks, total=len(file_tasks)):
        result = process_file(task, k)

        if result:
            write_header = not os.path.exists(K_neighbor_dist_feature_csv)
            df = pd.DataFrame(result, index=[0])
            df.to_csv(K_neighbor_dist_feature_csv, mode='a', header=write_header, index=False)


