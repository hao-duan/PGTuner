'''
Search for the true top-100 nearest neighbors of the query vectors in the given dataset.
'''

import os
import pandas as pd
from tqdm import tqdm
import time
from sklearn.neighbors import NearestNeighbors as NearestNeighbors_CPU
from utils.data_rw import read_bvecs, read_fvecs, save_ivecs

def process_search(args):
    subdir, file_text, root_dir, query_dir, groundtruth_dir = args

    subdir_path = os.path.join(root_dir, subdir)
    filename = os.path.splitext(file_text)[0]
    whole_name = subdir + '_' + filename
    filename_list = filename.split('_')
    dim = int(filename_list[2])

    file_path = os.path.join(subdir_path, file_text)

    if subdir in ['deep', 'glove', 'sift', 'paper', 'nytimes', 'crawl', 'msong', 'tiny', 'gist']:
        if dim == 128:
            query_path = os.path.join(query_dir, '{}/{}.bvecs'.format(subdir, dim))

            base_vectors = read_bvecs(file_path, None)
            query_vectors = read_bvecs(query_path, None)
        else:
            query_path = os.path.join(query_dir, '{}/{}.fvecs'.format(subdir, dim))

            base_vectors = read_fvecs(file_path, None)
            query_vectors = read_fvecs(query_path, None)
    else:
        query_path = os.path.join(query_dir, '{}/{}.fvecs'.format(subdir, filename))

        base_vectors = read_fvecs(file_path, None)
        query_vectors = read_fvecs(query_path, None)

    print('start searching for the true nearest neighbors...')
    nn = NearestNeighbors_CPU(n_neighbors=100, algorithm='brute', metric='euclidean')
    nn.fit(base_vectors)

    t1 = time.time()
    _, indices = nn.kneighbors(query_vectors)
    t2 = time.time()

    search_time = t2 - t1
    del base_vectors, query_vectors, nn

    save_path = os.path.join(groundtruth_dir, '{}/{}.ivecs'.format(subdir, filename))
    save_ivecs(indices, save_path)

    time_data = {"FileName": whole_name, "SearchTime": search_time}
    return time_data


if __name__ == '__main__':
    root_dir = "./Data/Base"
    query_dir = "./Data/Query"
    groundtruth_dir = "./Data/GroundTruth"
    bruteforce_search_time_csv = "./Data/bruteforce_search_time.csv"

    os.makedirs(groundtruth_dir, exist_ok=True)

    if os.path.exists(bruteforce_search_time_csv):
        df = pd.read_csv(bruteforce_search_time_csv, sep=',', header=0)
        exist_FileName = df['FileName'].tolist()
    else:
        exist_FileName = []

    file_tasks = []

    for subdir in tqdm(os.listdir(root_dir), total=len(os.listdir(root_dir))):
        # Replace the dataset names in the list with the datasets from which you need to search for the true nearest neighbors.
        if subdir in ['gist']:
            os.makedirs(os.path.join(groundtruth_dir, subdir), exist_ok=True)

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

                        if 1e6 <= size < 1e7 and whole_name not in exist_FileName:
                            args = (subdir, file_text, root_dir, query_dir, groundtruth_dir)
                            file_tasks.append(args)

    for task in tqdm(file_tasks, total=len(file_tasks)):
        result = process_search(task)

        if result:
            write_header = not os.path.exists(bruteforce_search_time_csv)
            df = pd.DataFrame(result, index=[0])
            df.to_csv(bruteforce_search_time_csv, mode='a', header=write_header, index=False)








