'''
DR feature extraction for the QD scenario.
'''

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import cupy as cp
from cuml.neighbors import NearestNeighbors
from utils.data_rw import read_bvecs, read_fvecs

def uniform_sample(max_num, sample_size):
    indices = np.random.choice(max_num, sample_size, replace=False)
    return indices

def get_q_k_neighbor_dists(b_vectors, q_vectors):
    b_vectors = cp.array(b_vectors)
    q_vectors = cp.array(q_vectors)

    nn = NearestNeighbors(n_neighbors=1000, algorithm='brute', metric='euclidean')
    nn.fit(b_vectors)

    distances, _ = nn.kneighbors(q_vectors)

    top_10_distances = distances[:, :10]
    other_distances = distances[:, 10:]

    mean_top_10_distances = top_10_distances.mean(axis=1)
    mean_other_distances = other_distances.mean(axis=1)

    zero_rows = cp.isclose(mean_other_distances, 0)
    indices_to_remove = cp.where(zero_rows)[0]

    mean_other_distances_non_zero = mean_other_distances[~zero_rows]
    mean_top_10_distances_non_zero = mean_top_10_distances[~zero_rows]

    ratios = mean_top_10_distances_non_zero / mean_other_distances_non_zero

    min_ratio = ratios.min()
    max_ratio = ratios.max()
    mean_ratio = ratios.mean()
    std_ratio = ratios.std()
    median_ratio = cp.percentile(ratios, 50)

    return min_ratio, median_ratio, max_ratio, mean_ratio,  std_ratio

def process_file(args):
    subdir, file_text, b_root_dir,  q_root_dir = args

    b_subdir_path = os.path.join(b_root_dir, subdir)
    q_subdir_path = os.path.join(q_root_dir, subdir)

    filename = os.path.splitext(file_text)[0]
    whole_name = subdir + '_' + filename

    filename_list = filename.split('_')
    level = int(filename_list[0])
    num = float(filename_list[1])
    dim = int(filename_list[2])
    size = int(pow(10, level) * 100000 * num)

    b_file_path = os.path.join(b_subdir_path, file_text)
    print(whole_name)

    whole_name1 = whole_name + '_25'
    whole_name2 = whole_name + '_50'
    whole_name3 = whole_name + '_75'
    whole_name4 = whole_name + '_100'

    if dim == 128 and subdir == 'sift':
        q_file_path1 = os.path.join(q_subdir_path, filename_list[2]+'_25'+'.bvecs')
        q_file_path2 = os.path.join(q_subdir_path, filename_list[2]+'_50'+'.bvecs')
        q_file_path3 = os.path.join(q_subdir_path, filename_list[2]+'_75'+'.bvecs')
        q_file_path4 = os.path.join(q_subdir_path, filename_list[2]+'_100'+'.bvecs')

        b_vectors = read_bvecs(b_file_path, None)
        q_vectors1 = read_bvecs(q_file_path1, None)
        q_vectors2 = read_bvecs(q_file_path2, None)
        q_vectors3 = read_bvecs(q_file_path3, None)
        q_vectors4 = read_bvecs(q_file_path4, None)

    else:
        q_file_path1 = os.path.join(q_subdir_path, filename_list[2]+'_25'+'.fvecs')
        q_file_path2 = os.path.join(q_subdir_path, filename_list[2]+'_50'+'.fvecs')
        q_file_path3 = os.path.join(q_subdir_path, filename_list[2]+'_75'+'.fvecs')
        q_file_path4 = os.path.join(q_subdir_path, filename_list[2]+'_100'+'.fvecs')

        b_vectors = read_fvecs(b_file_path, None)
        q_vectors1 = read_fvecs(q_file_path1, None)
        q_vectors2 = read_fvecs(q_file_path2, None)
        q_vectors3 = read_fvecs(q_file_path3, None)
        q_vectors4 = read_fvecs(q_file_path4, None)

    if b_vectors.dtype != np.float32:
        b_vectors = b_vectors.astype(np.float32)
        q_vectors1 = q_vectors1.astype(np.float32)
        q_vectors2 = q_vectors2.astype(np.float32)
        q_vectors3 = q_vectors3.astype(np.float32)
        q_vectors4 = q_vectors4.astype(np.float32)

    q_size = int(q_vectors1.shape[0])

    q_vectors_lis = [q_vectors1, q_vectors2, q_vectors3, q_vectors4]
    feature_data_lis = []

    print('Start extracting the DR features...')
    for i in tqdm(range(len(q_vectors_lis)), total = len(q_vectors_lis)):
        q_vectors = q_vectors_lis[i]
        t1 = time.time()

        min_ratio, median_ratio, max_ratio, mean_ratio,  std_ratio, = get_q_k_neighbor_dists(b_vectors, q_vectors)
        
        t2 = time.time()
        search_time = t2 - t1
    
        feature_data = {
            "FileName": whole_name1, "q_SIZE": q_size, "q_K_MinRatio": min_ratio, "q_K_MaxRatio": max_ratio, "q_K_MeanRatio": mean_ratio, "q_K_StdRatio": std_ratio,
            "q_SearchTime": search_time}
        feature_data_lis.append(feature_data)

    feature_data_lis[1]["FileName"]= whole_name2
    feature_data_lis[2]["FileName"]= whole_name3
    feature_data_lis[3]["FileName"]= whole_name4

    return feature_data_lis


if __name__ == '__main__':
    b_root_dir = "./Data/Base"
    q_root_dir = "./Data/Query"
    query_K_neighbor_dist_feature_csv = "./Data/query_K_neighbor_dist_ratio_feature.csv"

    exist_name = []
    if os.path.exists(query_K_neighbor_dist_feature_csv):
        df = pd.read_csv(query_K_neighbor_dist_feature_csv, sep=',', header=0)
        exist_name = list(df['FileName'])

    file_tasks = []

    for subdir in os.listdir(b_root_dir):
        # Replace the dataset names in the list with the datasets from which you need to extract the DR features.
        if subdir in ['gist']:
            subdir_path = os.path.join(b_root_dir, subdir)
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

                        if 1e5 <= size < 3e6 and dim in [96, 100, 128, 200, 256, 300, 384, 420, 960] and len(filename_list) == 3:
                            args = (subdir, file_text, b_root_dir, q_root_dir)
                            file_tasks.append(args)

    for task in tqdm(file_tasks, total=len(file_tasks)):
        feature_data_lis = process_file(task)
        for result in feature_data_lis:
            if result:
                write_header = not os.path.exists(query_K_neighbor_dist_feature_csv)
                df = pd.DataFrame(result, index=[0])
                df.to_csv(query_K_neighbor_dist_feature_csv, mode='a', header=write_header, index=False)


