import os
import pandas as pd
from tqdm import tqdm
import time
from sklearn.neighbors import NearestNeighbors as NearestNeighbors_CPU
from utils.data_rw import read_bvecs, read_fvecs, save_ivecs

def process_search(args):
    subdir, file_text, root_dir, query_dir, groundtruth_dir  = args

    subdir_path = os.path.join(root_dir, subdir)
    filename = os.path.splitext(file_text)[0]
    whole_name = subdir + '_' + filename

    filename_list = filename.split('_')
    dim = int(filename_list[2])
    
    file_path = os.path.join(subdir_path, file_text)

    filename1 = filename + '_25'
    filename2 = filename + '_50'
    filename3 = filename + '_75'
    filename4 = filename + '_100'

    indice_path1 = os.path.join(groundtruth_dir, '{}/{}.ivecs'.format(subdir, filename1))
    indice_path2 = os.path.join(groundtruth_dir, '{}/{}.ivecs'.format(subdir, filename2))
    indice_path3 = os.path.join(groundtruth_dir, '{}/{}.ivecs'.format(subdir, filename3))
    indice_path4 = os.path.join(groundtruth_dir, '{}/{}.ivecs'.format(subdir, filename4))

    indice_path_lis = [indice_path1, indice_path2, indice_path3, indice_path4]

    if dim == 128:
        query_path1 = os.path.join(query_dir, '{}/{}_25.bvecs'.format(subdir, dim))
        query_path2 = os.path.join(query_dir, '{}/{}_50.bvecs'.format(subdir, dim))
        query_path3 = os.path.join(query_dir, '{}/{}_75.bvecs'.format(subdir, dim))
        query_path4 = os.path.join(query_dir, '{}/{}_100.bvecs'.format(subdir, dim))

        base_vectors = read_bvecs(file_path, None)
        query_vectors1 = read_bvecs(query_path1, None)
        query_vectors2 = read_bvecs(query_path2, None)
        query_vectors3 = read_bvecs(query_path3, None)
        query_vectors4 = read_bvecs(query_path4, None)

    else:
        query_path1 = os.path.join(query_dir, '{}/{}_25.fvecs'.format(subdir, dim))
        query_path2 = os.path.join(query_dir, '{}/{}_50.fvecs'.format(subdir, dim))
        query_path3 = os.path.join(query_dir, '{}/{}_75.fvecs'.format(subdir, dim))
        query_path4 = os.path.join(query_dir, '{}/{}_100.fvecs'.format(subdir, dim))

        base_vectors = read_fvecs(file_path, None)
        query_vectors1 = read_fvecs(query_path1, None)
        query_vectors2 = read_fvecs(query_path2, None)
        query_vectors3 = read_fvecs(query_path3, None)
        query_vectors4 = read_fvecs(query_path4, None)
    
    query_vectors_lis = [query_vectors1, query_vectors2, query_vectors3, query_vectors4]

    nn = NearestNeighbors_CPU(n_neighbors=100, algorithm='brute', metric='euclidean')
    nn.fit(base_vectors)

    print('start searching for the true nearest neighbors...')
    t1 = time.time()

    for i in range(len(query_vectors_lis)):
        query_vectors = query_vectors_lis[i]
        _, indices = nn.kneighbors(query_vectors)

        del query_vectors

        save_path = indice_path_lis[i]
        save_ivecs(indices, save_path)

    t2 = time.time()
    search_time = t2 - t1

    time_data = {"FileName": whole_name, "SearchTime": search_time}
    return time_data

if __name__=='__main__':
    root_dir = "./Data/Base"
    query_dir = "./Data/Query"
    groundtruth_dir = "./Data/GroundTruth"
    bruteforce_search_time_csv = "./Data/bruteforce_search_time_qd_change.csv"

    if os.path.exists(bruteforce_search_time_csv):
        df = pd.read_csv(bruteforce_search_time_csv, sep=',', header=0)
        exist_FileName = df['FileName'].tolist()
    else:
        exist_FileName = []

    file_tasks = []
    for subdir in tqdm(os.listdir(root_dir), total = len(os.listdir(root_dir))):
        # Replace the dataset names in the list with the datasets from which you need to search for the true nearest neighbors.
        if subdir in ['gist']:
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

                        if 1e6 <= size < 2e6:
                            args = (subdir, file_text, root_dir, query_dir, groundtruth_dir)
                            file_tasks.append(args)

    for task in tqdm(file_tasks, total = len(file_tasks)):
        result = process_search(task)

        if result:
            write_header = not os.path.exists(bruteforce_search_time_csv)
            df = pd.DataFrame(result, index=[0])
            df.to_csv(bruteforce_search_time_csv, mode='a', header=write_header, index=False)


 





