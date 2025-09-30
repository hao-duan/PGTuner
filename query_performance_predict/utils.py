import numpy as np
import torch
import pandas as pd
import random
from torch.utils.data import Dataset
from Args import args

'''
----------------------read data---------------------------------
'''
def read_data_new(df):
    df_f = df[['FileName', 'efConstruction', 'M', 'efSearch', 'SIZE', 'q_SIZE', 'DIM', 'LID', 'Sum_K_MinDist', 'Sum_K_MaxDist', 'Sum_K_StdDist', 'q_K_MinRatio', 'q_K_MeanRatio','q_K_MaxRatio', 'q_K_StdRatio']]
    df_p = df[['recall', 'average_construct_dc_counts', 'average_search_dc_counts']]
    return df_f, df_p

def read_data_new_nsg(df):
    df_f = df[['FileName', 'K', 'L', 'L_nsg_C', 'R_nsg', 'C', 'L_nsg_S', 'SIZE', 'q_SIZE', 'DIM', 'LID', 'Sum_K_MinDist', 'Sum_K_MaxDist', 'Sum_K_StdDist', 'q_K_MinRatio', 'q_K_MeanRatio','q_K_MaxRatio', 'q_K_StdRatio']]
    df_p = df[['recall', 'average_NSG_s_dc_counts']]
    return df_f, df_p

'''
----------------------split the data into training, validation, and test sets---------------------------------
'''
def get_dataset(file_path):
    def split_df(group, train_frac=0.8, val_frac=0.1):
        shuffled_indices = np.random.permutation(len(group))
        train_end = int(len(group) * train_frac)
        val_end = int(len(group) * (train_frac + val_frac))

        if train_end % 2 == 0:
            train_indices = shuffled_indices[:train_end]
            val_indices = shuffled_indices[train_end:val_end]
        else:
            train_indices = shuffled_indices[:train_end+1]
            val_indices = shuffled_indices[train_end+1:val_end]
        test_indices = shuffled_indices[val_end:]

        return {
            'train': group.iloc[train_indices],
            'valid': group.iloc[val_indices],
            'test': group.iloc[test_indices]
        }

    df = pd.read_csv(file_path, sep=',', header=0)
    result = split_df(df)

    df_train = result['train']
    df_valid = result['valid']
    df_test = result['test']

    return df_train, df_valid, df_test

def split_data(df):
    def split_df(group, train_frac=0.8, val_frac=0.1):
        shuffled_indices = np.random.permutation(len(group))
        train_end = int(len(group) * train_frac)
        val_end = int(len(group) * (train_frac + val_frac))

        if train_end % 2 == 0:
            train_indices = shuffled_indices[:train_end]
            val_indices = shuffled_indices[train_end:val_end]
        else:
            train_indices = shuffled_indices[:train_end+1]
            val_indices = shuffled_indices[train_end+1:val_end]
        test_indices = shuffled_indices[val_end:]

        return {
            'train': group.iloc[train_indices],
            'valid': group.iloc[val_indices],
            'test': group.iloc[test_indices]
        }

    result = split_df(df)

    df_train = result['train']
    df_valid = result['valid']
    df_test = result['test']

    return df_train, df_valid, df_test

'''
----------------------dataFrame to numpy, numpy to tensor---------------------------------
'''
def df2np(df):
    np_data = df.to_numpy()
    return np_data

def np2ts(np_data):
    ts_data = torch.tensor(np_data).to(torch.float32)
    return ts_data


'''
----------------------CustomDataset class---------------------------------
'''
class CustomDataset(Dataset):
    def __init__(self, tensor1, tensor2):
        self.tensor1 = tensor1
        self.tensor2 = tensor2

    def __len__(self):
        return len(self.tensor1)  #

    def __getitem__(self, index):
        return self.tensor1[index], self.tensor2[index]


'''
----------------------data normalization class---------------------------------
'''
class Scaler_minmax_new_gpu:
    def __init__(self, num, device):
        self.num = num

        self.min = None
        self.max = None
        self.mean = None
        self.std = None

        if self.num == 6:
            self.min = torch.tensor([np.log10(20), 4, 1, 5, 2, 100], dtype=torch.float32).to(device)
            self.max = torch.tensor([np.log10(800), 100, torch.log10(torch.tensor(5000)), 8, 4, 1000], dtype=torch.float32).to(device)
        elif self.num == 3:
            self.min = torch.tensor([5, 2, 100], dtype=torch.float32).to(device)
            self.max = torch.tensor([8, 4, 1000], dtype=torch.float32).to(device)
        else:
            self.min = torch.tensor([0, 2, 2], dtype=torch.float32).to(device)
            self.max = torch.tensor([1, 5, torch.log10(torch.tensor(500000))], dtype=torch.float32).to(device)

    def fit(self, data):
        if self.num != 0:
            self.mean = torch.mean(data[:, self.num:], dim=0)
            self.std = torch.std(data[:, self.num:], dim=0)

    def transform(self, data):
        if self.num != 0:
            head_data = (data[:, :self.num] - self.min) / (self.max - self.min)
            tail_data = (data[:, self.num:] - self.mean) / self.std

            normalized_data = torch.cat((head_data, tail_data), dim=1)
        else:
            normalized_data = (data - self.min) / (self.max - self.min)
        return normalized_data

    def inverse_transform(self, data):
        if self.num != 0:
            head_data = data[:, :self.num] * (self.max - self.min) + self.min
            tail_data = data[:, self.num:] * self.std + self.mean

            raw_data = torch.cat((head_data, tail_data), dim=1)
        else:
            raw_data = data * (self.max - self.min) + self.min

        return raw_data

    def save_parameters(self, minmax_path, standard_path):
        # torch.save({'min': self.min, 'max': self.max}, minmax_path)
        torch.save({'mean': self.mean, 'std': self.std}, standard_path)

    def load_parameters(self, minmax_path, standard_path, device):
        # minmax_params = torch.load(minmax_path)
        # self.min = minmax_params['min']
        # self.max = minmax_params['max']

        standard_params = torch.load(standard_path)
        self.mean = standard_params['mean'].to(device)
        self.std = standard_params['std'].to(device)

class Scaler_minmax_new_gpu_nsg:
    def __init__(self, num, device):
        self.num = num

        self.min = None
        self.max = None
        self.mean = None
        self.std = None

        if self.num == 9:
            self.min = torch.tensor([100, 100, 150, 5, 300, 1, 5, 2, 100], dtype=torch.float32).to(device)
            self.max = torch.tensor([400, 400, 350, 90, 600, torch.log10(torch.tensor(1500)), 6, 4, 1000],
                                    dtype=torch.float32).to(device)
        elif self.num == 3:
            self.min = torch.tensor([5, 2, 100], dtype=torch.float32).to(device)
            self.max = torch.tensor([6, 4, 1000], dtype=torch.float32).to(device)
        else:
            self.min = torch.tensor([0, 1], dtype=torch.float32).to(device)
            self.max = torch.tensor([1, torch.log10(torch.tensor(50000))], dtype=torch.float32).to(device)

    def fit(self, data):
        if self.num != 0:
            self.mean = torch.mean(data[:, self.num:], dim=0)
            self.std = torch.std(data[:, self.num:], dim=0) + 1e-8

    def transform(self, data):
        if self.num != 0:
            head_data = (data[:, :self.num] - self.min) / (self.max - self.min)
            tail_data = (data[:, self.num:] - self.mean) / self.std

            normalized_data = torch.cat((head_data, tail_data), dim=1)
        else:
            normalized_data = (data - self.min) / (self.max - self.min)
        return normalized_data

    def inverse_transform(self, data):
        if self.num != 0:
            head_data = data[:, :self.num] * (self.max - self.min) + self.min
            tail_data = data[:, self.num:] * self.std + self.mean

            raw_data = torch.cat((head_data, tail_data), dim=1)
        else:
            raw_data = data * (self.max - self.min) + self.min

        return raw_data

    def save_parameters(self, minmax_path, standard_path):
        # torch.save({'min': self.min, 'max': self.max}, minmax_path)
        torch.save({'mean': self.mean, 'std': self.std}, standard_path)

    def load_parameters(self, minmax_path, standard_path, device):
        # minmax_params = torch.load(minmax_path)
        # self.min = minmax_params['min']
        # self.max = minmax_params['max']

        standard_params = torch.load(standard_path)
        self.mean = standard_params['mean'].to(device)
        self.std = standard_params['std'].to(device)

'''
----------------------load and save model--------------------------------
'''
def save_model(model, optimizer, epoch, file_path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, file_path)

def load_model(model, optimizer, file_path):
    checkpoint = torch.load(file_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    return model, optimizer, epoch

'''
----------------------calculate prediction errors---------------------------------
'''
def calculate_errors(performances_test, predicted_performances):
    errors = torch.abs(performances_test - predicted_performances)
    mean_errors = torch.mean(errors, dim=0)

    errors_percent = errors / ((performances_test + predicted_performances) / 2)
    mean_errors_percent = torch.mean(errors_percent, dim=0)

    qerrors = torch.max(performances_test / predicted_performances, predicted_performances / performances_test)
    mean_qerrors = torch.mean(qerrors, dim=0)

    return mean_errors, mean_errors_percent, mean_qerrors



