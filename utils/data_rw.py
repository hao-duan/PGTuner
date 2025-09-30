import sys
sys.path.append('./utils')
import struct
import numpy as np

# read high-dimensional vector data from a bvecs file (for SIFT dataset)
def read_bvecs(file_path, num=None):
    vectors = []
    with open(file_path, 'rb') as f:
        count = 0
        while True:
            if num is not None and count >= num:
                break
            dim_bytes = f.read(4)
            if not dim_bytes:
                break
            dim, = struct.unpack('I', dim_bytes)
            vector_bytes = f.read(dim)
            vector = np.frombuffer(vector_bytes, dtype=np.uint8)
            vectors.append(vector)
            count += 1
    return np.array(vectors)

# read high-dimensional vector data from a fvecs file
def read_fvecs(file_path, num=None):
    vectors = []
    count = 0
    with open(file_path, 'rb') as f:
        while True:
            if num is not None and count >= num:
                break
            k_bytes = f.read(4)
            if not k_bytes:
                break
            k, = struct.unpack('I', k_bytes)
            vector_bytes = f.read(k * 4)  # For fvecs, each dimension is a float (4 bytes)
            vector = np.frombuffer(vector_bytes, dtype=np.float32)
            vectors.append(vector)
            count += 1
    return np.array(vectors)

# read the true nearest neighbors from an ivecs file.
def read_ivecs(file_path):
    indices = []
    with open(file_path, 'rb') as f:
        while True:
            k_bytes = f.read(4)
            if not k_bytes:
                break
            k, = struct.unpack('I', k_bytes)
            vector_bytes = f.read(k * 4)  # For ivecs, each dimension is an int (4 bytes)
            indice = np.frombuffer(vector_bytes, dtype=np.int32)
            indices.append(indice)
    return np.array(indices)

# store high-dimensional vector data into a bvecs file (for SIFT dataset)
def save_bvecs(vectors, file_path):
    with open(file_path, 'wb') as f:
        dim = vectors.shape[1]
        for vector in vectors:
            f.write(struct.pack('I', dim))
            f.write(vector.tobytes())

# store high-dimensional vector data into a fvecs file
def save_fvecs(vectors, file_path):
    dim = vectors.shape[1]
    with open(file_path, 'wb') as f:
        for vector in vectors:
            # 获取向量的维度
            # 写入维度信息
            f.write(struct.pack('I', dim))
            # 写入向量数据
            f.write(vector.astype(np.float32).tobytes())

# store the true nearest neighbors into an ivecs file
def save_ivecs(indices, file_path):
    dim = indices.shape[1]
    with open(file_path, 'wb') as f:
        for indice in indices:
            # 获取向量的维度
            # 写入维度信息
            f.write(struct.pack('I', dim))
            # 写入向量数据
            f.write(indice.astype(np.int32).tobytes())















