import sys
sys.path.append('./utils')
import numpy as np
import struct

def uniform_sample(max_num, sample_size):
    indices = np.random.choice(max_num, sample_size, replace=False)
    return indices

def normal_sample(max_num, sample_size, mean=0.5, std=0.1):
    indices = (np.random.normal(mean, std, sample_size) * max_num).astype(int)
    indices = np.clip(indices, 0, max_num-1)
    return indices

# random sampling from the original vector dataset (for SIFT dataset)
def data_sample_random(file_path, dim, max_num, sample_size, flag=0):
    if flag == 0:
        indices = uniform_sample(max_num, sample_size)
    else:
        indices = normal_sample(max_num, sample_size, mean=0.5, std=0.1)

    byte_per_vector = 4 + dim
    sampled_vectors = []

    with open(file_path, 'rb') as f:
        for id in indices:
            f.seek(id * byte_per_vector)
            dim_bytes = f.read(4)
            dim, = struct.unpack('I', dim_bytes)
            vector_bytes = f.read(dim)
            vector = np.frombuffer(vector_bytes, dtype=np.uint8)
            sampled_vectors.append(vector)

    return np.array(sampled_vectors)

# random sampling from the original vector dataset
def data_sample_random_float(file_path, dim, max_num, sample_size, flag=0):
    if flag == 0:
        indices = uniform_sample(max_num, sample_size)
    else:
        indices = normal_sample(max_num, sample_size, mean=0.5, std=0.1)

    byte_per_vector = 4 + 4 * dim
    sampled_vectors = []

    with open(file_path, 'rb') as f:
        # for id in tqdm(indices, total = len(indices)):
        for id in indices:
            f.seek(id * byte_per_vector)
            dim_bytes = f.read(4)
            dim, = struct.unpack('I', dim_bytes)
            vector_bytes = f.read(4 * dim)
            vector = np.frombuffer(vector_bytes, dtype=np.float32)
            sampled_vectors.append(vector)

    return np.array(sampled_vectors)

# sequential sampling from the original vector dataset (for SIFT dataset)
def data_sample_sequential(file_path, dim, max_num, sample_size, start_id):
    byte_per_vector = 4 + dim
    vectors = []

    end_id = start_id + sample_size

    with open(file_path, 'rb') as f:
        if end_id <= max_num:
            f.seek(start_id * byte_per_vector)
            for _ in range(sample_size):
                dim_bytes = f.read(4)
                dim_, = struct.unpack('I', dim_bytes)
                vector_bytes = f.read(dim_)
                vector = np.frombuffer(vector_bytes, dtype=np.uint8)
                vectors.append(vector)
        else:
            f.seek(start_id * byte_per_vector)
            for _ in range(max_num - start_id):
                dim_bytes = f.read(4)
                dim, = struct.unpack('I', dim_bytes)
                vector_bytes = f.read(dim)
                vector = np.frombuffer(vector_bytes, dtype=np.uint8)
                vectors.append(vector)
            f.seek(0)
            for _ in range(end_id - max_num):
                dim_bytes = f.read(4)
                dim, = struct.unpack('I', dim_bytes)
                vector_bytes = f.read(dim)
                vector = np.frombuffer(vector_bytes, dtype=np.uint8)
                vectors.append(vector)

    return np.array(vectors)

# sequential sampling from the original vector dataset
def data_sample_sequential_float(file_path, dim, max_num, sample_size, start_id):
    byte_per_vector = 4 + 4 * dim
    vectors = []

    end_id = start_id + sample_size

    with open(file_path, 'rb') as f:
        if end_id <= max_num:
            f.seek(start_id * byte_per_vector)
            for _ in range(sample_size):
                k_bytes = f.read(4)
                if not k_bytes:
                    break
                k, = struct.unpack('I', k_bytes)
                vector_bytes = f.read(k * 4)
                vector = np.frombuffer(vector_bytes, dtype=np.float32)
                vectors.append(vector)
        else:
            f.seek(start_id * byte_per_vector)
            for _ in range(max_num - start_id):
                k_bytes = f.read(4)
                if not k_bytes:
                    break
                k, = struct.unpack('I', k_bytes)
                vector_bytes = f.read(k * 4)
                vector = np.frombuffer(vector_bytes, dtype=np.float32)
                vectors.append(vector)
            f.seek(0)
            for _ in range(end_id - max_num):
                k_bytes = f.read(4)
                if not k_bytes:
                    break
                k, = struct.unpack('I', k_bytes)
                vector_bytes = f.read(k * 4)
                vector = np.frombuffer(vector_bytes, dtype=np.float32)
                vectors.append(vector)

    return np.array(vectors)






