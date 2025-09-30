'''
Sample subsets from the given vector dataset through random sampling or sequential sampling.
'''

import os
import numpy as np
from tqdm import tqdm
import random
from utils.data_sample import data_sample_random, data_sample_random_float, data_sample_sequential, data_sample_sequential_float
from utils.data_rw import save_bvecs, save_fvecs

def process_sample_task(args):
    b_path, dim, max_num, size, dtname, level, num, i, current_directory, seed = args

    np.random.seed(seed)
    random.seed(seed)

    start_id = np.random.randint(0, max_num)

    if dim == 128:
        if size >= 1 * 1e6:
            vectors = data_sample_sequential(b_path, dim, max_num, size, start_id)
            save_path = os.path.join(current_directory, 'Data/Base/{}/{}_{}_{}_{}.bvecs'.format(dtname, level, num, dim, i))
            save_bvecs(vectors, save_path)
        else:
            vectors = data_sample_random(b_path, dim, max_num, size, flag=0)
            save_path = os.path.join(current_directory, 'Data/Base/{}/{}_{}_{}_{}.bvecs'.format(dtname, level, num, dim, i))
            save_bvecs(vectors, save_path)
    else:
        if size >= 1 * 1e6:
            vectors = data_sample_sequential_float(b_path, dim, max_num, size, start_id)
            save_path = os.path.join(current_directory, 'Data/Base/{}/{}_{}_{}_{}.fvecs'.format(dtname, level, num, dim, i))
            save_fvecs(vectors, save_path)
        else:
            vectors = data_sample_random_float(b_path, dim, max_num, size, flag=0)
            save_path = os.path.join(current_directory, 'Data/Base/{}/{}_{}_{}_{}.fvecs'.format(dtname, level, num, dim, i))
            save_fvecs(vectors, save_path)

def sample_main(current_directory, dim2dtname, dim_list, sample_seed_list):
    exist_group = []
    numbers = [1]

    tasks = []
    for base_num in [1e5]:
        level = {1e4: -1, 1e5: 0, 1e6: 1, 1e7: 2}.get(base_num, 3)

        for num in numbers:
            size = int(num * base_num)
            for dim in dim_list:
                dim = int(dim)

                if dim in dim2dtname.keys():
                    if (size, dim) not in exist_group:
                        dtname, max_num = dim2dtname[dim]
                        max_num = int(max_num)

                        if size < max_num:
                            if dim == 128:
                                b_path = os.path.join(current_directory, 'Data/{}-{}/{}_base.bvecs'.format(dtname, dim, dtname))
                            else:
                                b_path = os.path.join(current_directory, 'Data/{}-{}/{}_base.fvecs'.format(dtname, dim, dtname))

                            for i, seed in enumerate(sample_seed_list):
                                if i in [1]:
                                    tasks.append((b_path, dim, max_num, size, dtname, level, num, i, current_directory, seed))

    for task in tqdm(tasks, total=len(tasks)):
        process_sample_task(task)

if __name__ == '__main__':
    current_directory = os.getcwd()

    root_dir = "./Data/Base"

    sample_seed_list = [21, 42]
    dim_list = [96]
    dim2dtname = {96:['deep', 100000], 100:['glove', 1183514], 128:['sift', 5e7], 200:['paper', 2029997],
                  300:['crawl', 1989995], 420:['msong', 992272], 960:['gist', 1000000]}

    sample_main(current_directory, dim2dtname, dim_list, sample_seed_list)
