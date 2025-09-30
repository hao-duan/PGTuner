import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import random
from utils.LID_estimate import intrinsic_dim
from utils.data_rw import read_bvecs, read_fvecs

def process_file(args):
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
        vectors = read_bvecs(file_path,None)
    else:
        vectors = read_fvecs(file_path, None)

    if vectors.dtype != np.float32:
        vectors = vectors.astype(np.float32)

    print('Start computing LID...')
    if size <= 1e6:
        t1 = time.time()
        lid = intrinsic_dim(vectors, 'MLE_NN')
        t2 = time.time()
        lid_time = t2 - t1
    else:
        t1 = time.time()
        lids = []

        if size < 1e7:
            sample_size = int(1e6)
            sample_num = int(size / sample_size + 0.5)
        else:
            sample_size = int(5e6)
            sample_num = int(size / 5e6 + 0.5)

        for i in tqdm(range(sample_num), total = sample_num):
            sample_indexs = random.sample(range(0, size), int(sample_size))
            sample_vectors = vectors[sample_indexs]
            lid = intrinsic_dim(sample_vectors, 'MLE_NN')
            lids.append(lid)

        lid = sum(lids) / sample_num

        t2 = time.time()
        lid_time = t2 - t1

    feature_data = {"FileName": whole_name, "SIZE": size, "DIM": dim, "LID": lid, "LIDTime": lid_time}
    return feature_data

if __name__=='__main__':
    root_dir = "./Data/Base"
    LID_feature_csv = "./Data/LID_feature.csv"

    exist_name = []
    if os.path.exists(LID_feature_csv):
        df = pd.read_csv(LID_feature_csv, sep=',', header=0)
        exist_name = list(df['FileName'])

    file_tasks = []
    for subdir in os.listdir(root_dir):
        # Replace the dataset names in the list with the datasets from which you need to calculate the LID.
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

                        if 1e6 <= size < 1e7 and dim in [96, 100, 128, 200, 256, 300, 384, 420, 960] and whole_name not in exist_name and len(filename_list) == 3:
                            args = (subdir, file_text, root_dir)
                            file_tasks.append(args)

    for task in tqdm(file_tasks, total=len(file_tasks)):
        result = process_file(task)
        if result:
            write_header = not os.path.exists(LID_feature_csv)
            df = pd.DataFrame(result, index=[0])
            df.to_csv(LID_feature_csv, mode='a', header=write_header, index=False)


