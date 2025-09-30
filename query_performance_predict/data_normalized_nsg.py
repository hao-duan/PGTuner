import os
import numpy as np
import pandas as pd
import torch
from utils import df2np, np2ts, Scaler_minmax_new_gpu_nsg

if __name__ == '__main__':
    use_cuda = torch.cuda.is_available()
    print(use_cuda)
    device = torch.device("cuda" if use_cuda else "cpu")

    current_directory = os.getcwd()
    parent_directory = os.path.dirname(current_directory)

    os.makedirs(os.path.join(current_directory, 'scaler_paras'), exist_ok=True)
    os.makedirs(os.path.join(current_directory, 'scaler_paras/NSG_KNNG'), exist_ok=True)
    feature_standard_path = os.path.join(current_directory, 'scaler_paras/NSG_KNNG/feature_standard.npz')

    train_data_path = os.path.join(parent_directory, 'NSG_KNNG/Data/train_data.csv')
    data_fetaure_path = os.path.join(parent_directory, 'NSG_KNNG/Data/ata_feature.csv')


    df_train = pd.read_csv(train_data_path, sep=',', header=0)
    df_f = df_train[['K', 'L', 'L_nsg_C', 'R_nsg', 'C', 'L_nsg_S', 'SIZE', 'q_SIZE', 'DIM', 'LID', 'Sum_K_MinDist', 'Sum_K_MaxDist', 'Sum_K_StdDist',
                     'q_K_MinRatio', 'q_K_MeanRatio','q_K_MaxRatio', 'q_K_StdRatio']]

    feature = df2np(df_f)
    feature[:, 5:8] = np.log10(feature[:, 5:8])

    feature = np2ts(feature).to(device)

    print('-------------------data normalization-------------------')
    feature_scaler = Scaler_minmax_new_gpu_nsg(9, device)
    if os.path.exists(feature_standard_path):
        feature_scaler.load_parameters(None, feature_standard_path, device)
    else:
        feature_scaler.fit(feature)
        feature_scaler.save_parameters(None, feature_standard_path)
