import sys
sys.path.append('./utils')
import os
import numpy as np
import random
from data_rw import read_bvecs, read_fvecs, save_bvecs, save_fvecs

if __name__ == '__main__':
    s = 20
    np.random.seed(s)
    random.seed(s)

    current_directory = os.getcwd()
    parent_directory = os.path.dirname(current_directory)

    dtname2dim = {'gist':960}
    dtname2size = {'gist': 1000}

    # Replace the dataset name with the dataset from which you need to collect data.
    dataset_name = 'gist'
    dim = dtname2dim[dataset_name]
    max_size = int(dtname2size[dataset_name])

    if not dataset_name == 'sift':
        init_q_path = os.path.join(parent_directory, 'Data/Query/{}/{}.fvecs'.format(dataset_name, dim))

        q_path1 = os.path.join(parent_directory, 'Data/Query/{}/{}_25.fvecs'.format(dataset_name, dim))
        q_path2 = os.path.join(parent_directory, 'Data/Query/{}/{}_50.fvecs'.format(dataset_name, dim))
        q_path3 = os.path.join(parent_directory, 'Data/Query/{}/{}_75.fvecs'.format(dataset_name, dim))
        q_path4 = os.path.join(parent_directory, 'Data/Query/{}/{}_100.fvecs'.format(dataset_name, dim))

        init_q_vectors = read_fvecs(init_q_path, num=None)
    else:
        init_q_path = os.path.join(parent_directory, 'Data/Query/{}/{}.bvecs'.format(dataset_name, dim))

        q_path1 = os.path.join(parent_directory, 'Data/Query/{}/{}_25.bvecs'.format(dataset_name, dim))
        q_path2 = os.path.join(parent_directory, 'Data/Query/{}/{}_50.bvecs'.format(dataset_name, dim))
        q_path3 = os.path.join(parent_directory, 'Data/Query/{}/{}_75.bvecs'.format(dataset_name, dim))
        q_path4 = os.path.join(parent_directory, 'Data/Query/{}/{}_100.bvecs'.format(dataset_name, dim))

        init_q_vectors = read_bvecs(init_q_path, num=None)

    temp_index = init_q_vectors.shape[0]
    half_dim = int(dim / 2)

    index1 = int(temp_index*0.25)
    index2 = int(temp_index*0.5)
    index3 = int(temp_index*0.75)

    indices1 = np.random.choice(max_size, index1, replace=False)
    indices2 = np.random.choice(max_size, index2, replace=False)
    indices3 = np.random.choice(max_size, index3, replace=False)

    q_vectors1 = init_q_vectors.copy()
    q_vectors2 = init_q_vectors.copy()
    q_vectors3 = init_q_vectors.copy()

    #gist
    if dataset_name == 'gist':
        mean=0.1
        std = 0.1
    elif dataset_name == 'sift':
        mean = 127.5
        std = 25

    if not dataset_name == 'sift':
        q_vectors1[indices1] = init_q_vectors[indices1] + np.clip(np.random.normal(mean, std, size=(index1, dim)), 0, 10)
        q_vectors2[indices2] = init_q_vectors[indices2] + np.clip(np.random.normal(mean, std, size=(index2, dim)), 0, 10)
        q_vectors3[indices3] = init_q_vectors[indices3] + np.clip(np.random.normal(mean, std, size=(index3, dim)), 0, 10)
        q_vectors4 = init_q_vectors + np.clip(np.random.normal(mean, std, size=(temp_index, dim)), 0, 10)

        save_fvecs(q_vectors1, q_path1)
        save_fvecs(q_vectors2, q_path2)
        save_fvecs(q_vectors3, q_path3)
        save_fvecs(q_vectors4, q_path4)
    else:
        q_vectors1[indices1] = init_q_vectors[indices1] + np.clip(np.random.normal(mean, std, size=(index1, dim)), 0, 255).astype(int)
        q_vectors2[indices2] = init_q_vectors[indices2] + np.clip(np.random.normal(mean, std, size=(index2, dim)), 0, 255).astype(int)
        q_vectors3[indices3] = init_q_vectors[indices3] + np.clip(np.random.normal(mean, std, size=(index3, dim)), 0, 255).astype(int)
        q_vectors4 = init_q_vectors + np.clip(np.random.normal(mean, std, size=(temp_index, dim)), 0, 255).astype(int)

        save_bvecs(q_vectors1, q_path1)
        save_bvecs(q_vectors2, q_path2)
        save_bvecs(q_vectors3, q_path3)
        save_bvecs(q_vectors4, q_path4)
