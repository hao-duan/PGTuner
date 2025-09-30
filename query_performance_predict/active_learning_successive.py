import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
from torch.backends import cudnn
import random
from sklearn.neighbors import NearestNeighbors
import time
from tqdm import tqdm

from models import Direct_Predict_MLP
from trainer import dipredict_train
from utils import read_data_new, split_data, df2np, np2ts, CustomDataset, load_model, save_model, Scaler_minmax_new_gpu, calculate_errors
from Args import args

'''
This code is used for model transfer in successive transfer tuning across multiple datasets.
'''

def get_feature_vectors(model, input_feature):
    model.eval()

    with torch.no_grad():
        input_feature_vectors = model.get_feature_vectors(input_feature, 3)
        input_feature_vectors_l2 = F.normalize(input_feature_vectors, p=2, dim=1)

    return input_feature_vectors_l2

def get_nn_dist(labeled_feature_vectors, query_feature_vectors, k):  
    labeled_feature_vectors_np = labeled_feature_vectors.cpu().numpy()
    query_feature_vectors_np = query_feature_vectors.cpu().numpy()

    nn = NearestNeighbors(n_neighbors=k, algorithm='brute', metric='euclidean')
    nn.fit(labeled_feature_vectors_np)

    distances, _ = nn.kneighbors(query_feature_vectors_np)

    min_distances = distances[:, -1]
    min_distances_np = min_distances

    del distances, nn
    return min_distances_np

def get_input_feature(df_data_feature, df_config_unit):
    efSs = np.array(
        [10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320,
         340, 360, 380, 400, 420, 440, 460, 480, 500, 520, 540, 560, 580, 600, 620, 640,
         660, 680, 700, 730, 760, 790, 820, 850, 880, 910, 940, 970, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700,
         1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000, 3100,
         3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900, 4000, 4100, 4200, 4300, 4400, 4500, 4600, 4700, 4800, 4900,
         5000])

    df_config_whole = df_config_unit.loc[df_config_unit.index.repeat(len(efSs))].reset_index(drop=True)
    df_config_whole['efSearch'] = np.tile(efSs, len(df_config_unit))

    df_feature = pd.merge(df_data_feature, df_config_whole, on='FileName', how='right')

    return df_feature

def feature_df2feature_tensor(df_feature, device, feature_scaler):
    feature_raw = df2np(df_feature)

    feature_raw[:, 0] = np.log10(feature_raw[:, 0])
    feature_raw[:, 2:5] = np.log10(feature_raw[:, 2:5])

    feature_raw = np2ts(feature_raw).to(device)

    feature_tensor = feature_scaler.transform(feature_raw)

    del feature_raw
    return feature_tensor

def performance_df2performance_tensor(df_performance, device, performance_scaler):
    performance_raw = df2np(df_performance)

    performance_raw[:, 1:] = np.log10(performance_raw[:, 1:])

    performance_raw = np2ts(performance_raw).to(device)

    performance_tensor = performance_scaler.transform(performance_raw)

    del performance_raw
    return performance_tensor

def CoreSetSelecting(labeled_feature_vectors, query_feature_vectors, selected_num):
    min_distances = get_nn_dist(labeled_feature_vectors, query_feature_vectors, 1)
    min_distances = min_distances.reshape((-1, 94))
    min_distances = np.mean(min_distances, axis=1)

    min_index = np.argmin(min_distances)
    max_index = np.argmax(min_distances)

    mean_value = np.mean(min_distances )
    mean_index = np.argmin(np.abs(min_distances  - mean_value))

    if selected_num ==  1:
        selected_indices = [max_index]
    elif selected_num ==  2:
        selected_indices = [mean_index, max_index]
    else:
        selected_indices = [min_index, mean_index, max_index]

    return selected_indices


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    cudnn.enabled = False
    cudnn.benchmark = False
    cudnn.deterministic = True

    use_cuda = torch.cuda.is_available()
    print(use_cuda)
    device = torch.device("cuda" if use_cuda else "cpu")

    current_directory = os.getcwd() 
    parent_directory = os.path.dirname(current_directory) 

    efSs = np.array(
        [10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 220, 240, 260, 280, 300, 320,
         340, 360, 380, 400, 420, 440, 460, 480, 500, 520, 540, 560, 580, 600, 620, 640,
         660, 680, 700, 730, 760, 790, 820, 850, 880, 910, 940, 970, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700,
         1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000, 3100,
         3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900, 4000, 4100, 4200, 4300, 4400, 4500, 4600, 4700, 4800, 4900,
         5000])

    config_unit_data_path = "../Data/config_unit_data.csv"

    filename_dic = {'deep1': 'deep_1_1_96_1', 'sift1': 'sift_1_1_128_1', 'glove': 'glove_1_1.183514_100', 'paper': 'paper_1_2.029997_200',
                    'crawl': 'crawl_1_1.989995_300', 'msong': 'msong_0_9.92272_420', 'nytimes': 'nytimes_0_2.9_256', 'tiny': 'tiny_1_1_384',
                    'gist': 'gist_1_1.0_960', 'deep10': 'deep_2_1_96', 'sift50': 'sift_2_5_128_1',
                    'sift2': 'sift_1_2_128_1', 'sift3': 'sift_1_3_128_1', 'sift4': 'sift_1_4_128_1', 'sift5': 'sift_1_5_128_1',
                    'gist_25': 'gist_1_1.0_960_25', 'gist_50': 'gist_1_1.0_960_50', 'gist_75': 'gist_1_1.0_960_75', 'gist_100': 'gist_1_1.0_960_100'}

    max_selected_num = args.max_selected_num
    selected_num = 2
    selected_rounds = max_selected_num // selected_num

    last_dataset_name = args.last_dataset_name

    dataset_name = args.dataset_name
    filename = filename_dic[dataset_name]

    mode = args.experiment_mode
    if mode == 'dataset_change':
        init_test_data_path = os.path.join(parent_directory, 'Data/test_data_main.csv')
    elif mode == 'ds_change':
        init_test_data_path = os.path.join(parent_directory, 'Data/test_data_ds_change.csv')
    elif mode == 'qd_change':
        init_test_data_path = os.path.join(parent_directory, 'Data/test_data_qd_change.csv')

    if dataset_name == 'sift50':
        config_unit_data_path = "../Data/config_unit_data_sift.csv"
    else:
        config_unit_data_path = "../Data/config_unit_data.csv"

    data_fetaure_path = os.path.join(parent_directory, 'Data/data_feature.csv')

    os.makedirs('../Data/active_learning_data', exist_ok=True)
    os.makedirs('../Data/active_learning_data/{}'.format(mode), exist_ok=True)
    os.makedirs(os.path.join(current_directory, 'scaler_paras/{}'.format(mode)), exist_ok=True)
    os.makedirs(os.path.join(current_directory, 'model_checkpoints/{}'.format(mode)), exist_ok=True)

    if not last_dataset_name:
        init_train_data_path = os.path.join(parent_directory, 'Data/train_data.csv')
        init_model_save_path = os.path.join(current_directory, 'model_checkpoints/{}_{}_{}_{}_checkpoint.pth'.format(
                                                          args.dipredict_layer_sizes, args.dipredict_n_epochs,
                                                               args.dipredict_batch_size, args.dipredict_lr))
        init_feature_standard_path = os.path.join(current_directory, 'scaler_paras/feature_standard.npz')
    else:
        init_train_data_path = "../Data/active_learning_data/{}/{}_train_data_{}_{}.csv".format(mode, last_dataset_name, selected_num, selected_rounds)
        init_model_save_path = os.path.join(current_directory, 'model_checkpoints/{}/{}_{}_{}_{}_{}_{}_{}_checkpoint.pth'.format(
                                                mode, last_dataset_name, args.dipredict_layer_sizes, args.dipredict_n_epochs,
                                                     args.dipredict_batch_size, args.dipredict_lr, selected_num, selected_rounds))
        init_feature_standard_path = os.path.join(current_directory, 'scaler_paras/{}/{}_feature_standard_{}_{}.npz'.format(
                                                                mode, last_dataset_name, selected_num, selected_rounds))

    new_train_data_path = "../Data/active_learning_data/{}/{}_train_data_{}_{}.csv".format(mode, dataset_name, selected_num, selected_rounds)
    selected_config_path = "../Data/active_learning_data/{}/{}_selected_config_{}_{}.csv".format(mode, dataset_name, selected_num, selected_rounds)

    new_feature_standard_path = os.path.join(current_directory, 'scaler_paras/{}/{}_feature_standard_{}_{}.npz'.format(mode, dataset_name, selected_num, selected_rounds))
    new_model_save_path = os.path.join(current_directory, 'model_checkpoints/{}/{}_{}_{}_{}_{}_{}_{}_checkpoint.pth'.format(
                                                     mode, dataset_name, args.dipredict_layer_sizes, args.dipredict_n_epochs,
                                                          args.dipredict_batch_size, args.dipredict_lr, selected_num, selected_rounds))

    test_error_dic = {'rec_MAE': [], 'ct_dc_MAE': [], 'st_dc_MAE': [], 'rec_MAPE': [], 'ct_dc_MAPE': [], 'st_dc_MAPE': [], 'distance_threshold': [],
                      'mean_distance': [], 'duration_time': [], 'each_detect_time': []}
    test_error_path = "../Data/active_learning_data/{}/{}_test_error_{}_{}.csv".format(mode, dataset_name,  selected_num, selected_rounds)

    print('-------------------load the normalizer-------------------')
    feature_scaler = Scaler_minmax_new_gpu(6, device)
    feature_scaler.load_parameters(None, init_feature_standard_path, device)
    performance_scaler = Scaler_minmax_new_gpu(0, device)

    print('-------------------load model-------------------')
    dipredict_layer_sizes = eval(args.dipredict_layer_sizes)

    model = Direct_Predict_MLP(dipredict_layer_sizes)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.dipredict_lr, weight_decay=args.weight_decay)
    model, _, _ = load_model(model, optimizer, init_model_save_path)

    print('-------------------load the initial labeled data and generate feature vectors-------------------')
    df_data_feature = pd.read_csv(data_fetaure_path, sep=',', header=0)
    current_train_df = pd.read_csv(init_train_data_path, sep=',', header=0)

    selected_df_feature_labeled = current_train_df[
        ['efConstruction', 'M', 'efSearch', 'SIZE', 'q_SIZE', 'DIM', 'LID', 'Sum_K_MinDist', 'Sum_K_MaxDist',
         'Sum_K_StdDist', 'q_K_MinRatio', 'q_K_MeanRatio', 'q_K_MaxRatio',
         'q_K_StdRatio']]

    selected_labeled_feature = feature_df2feature_tensor(selected_df_feature_labeled, device, feature_scaler)
    selected_labeled_feature_vectors = get_feature_vectors(model, selected_labeled_feature)

    detected_df_feature_labeled = current_train_df[['efConstruction', 'M', 'efSearch', 'SIZE', 'q_SIZE', 'DIM', 'LID', 'Sum_K_MinDist', 'Sum_K_MaxDist',
                                                    'Sum_K_StdDist', 'q_K_MinRatio', 'q_K_MeanRatio', 'q_K_MaxRatio', 'q_K_StdRatio']]

    detected_labeled_feature = feature_df2feature_tensor(detected_df_feature_labeled, device, feature_scaler)
    detected_labeled_feature_vectors = get_feature_vectors(model, detected_labeled_feature)

    print('-------------------compute the distances between the feature vectors of labeled data and their nearest neighbors-------------------')
    min_distance = get_nn_dist(detected_labeled_feature_vectors, detected_labeled_feature_vectors, 2)

    distance_threshold = np.percentile(min_distance, 95)

    print('-------------------load the unlabeled data and generate feature vectors-------------------')
    df_data_feature_test = df_data_feature[df_data_feature['FileName'] == filename]

    df_config_unit = pd.read_csv(config_unit_data_path, sep=',', header=0)
    df_config_unit['FileName'] = filename

    df_feature_unlabeled = get_input_feature(df_data_feature_test, df_config_unit)

    df_feature_query = df_feature_unlabeled[
        ['efConstruction', 'M', 'efSearch', 'SIZE', 'q_SIZE', 'DIM', 'LID', 'Sum_K_MinDist', 'Sum_K_MaxDist',
         'Sum_K_StdDist', 'q_K_MinRatio', 'q_K_MeanRatio', 'q_K_MaxRatio', 'q_K_StdRatio']]

    query_feature = feature_df2feature_tensor(df_feature_query, device, feature_scaler)
    query_feature_vectors = get_feature_vectors(model, query_feature)

    print('-------------------compute the distances between the feature vectors of unlabeled data and their nearest neighbors among the feature vectors of labeled data-------------------')
    min_distance = get_nn_dist(detected_labeled_feature_vectors, query_feature_vectors, 1)
    mean_distance = np.mean(min_distance)

    dist_flag = (mean_distance <= distance_threshold)
    if dist_flag:
        print('The current unlabeled data is similar to the labeled data so that the pre-trained QPP model can be directly used.')
        feature_scaler.save_parameters(None, new_feature_standard_path)
        save_model(model, optimizer, args.dipredict_n_epochs, new_model_save_path)
    else:
        print('The current unlabeled data is not similar to the labeled data; start active learning...')
        real_data_test_df = pd.read_csv(init_test_data_path, sep=',', header=0)
        real_data_test_df = real_data_test_df[real_data_test_df['FileName'] == filename]

        current_detected_df_feature_labeled = detected_df_feature_labeled.copy()  # data of all labeled combinations used for updating similarity detection
        current_selected_df_feature_labeled = selected_df_feature_labeled.copy()  # all training input feature data used for updating configuration selection
        current_df_config_unlabeled = df_config_unit.copy()  # all remaining unlabeled efC–M combination data used for updating

        ts = time.time()
        for round_num in tqdm(range(selected_rounds), total=selected_rounds):
            print('-------------------select unlabeled data------------------')
            selected_indices = CoreSetSelecting(selected_labeled_feature_vectors, query_feature_vectors, selected_num)
            df_selected_config = current_df_config_unlabeled.iloc[selected_indices]

            if not os.path.exists(selected_config_path):
                df_selected_config.to_csv(selected_config_path, index=False, mode='w', header=True)
            else:
                df_selected_config.to_csv(selected_config_path, index=False, mode='a', header=False)

            # update the unlabeled efC-M combinations
            current_df_config_unlabeled = current_df_config_unlabeled.drop(selected_indices)
            current_df_config_unlabeled = current_df_config_unlabeled.reset_index(drop=True)

            # obtain the training data of the selected configuration, mix it with the existing training data, and retrain the model
            selected_data_df = pd.merge(real_data_test_df, df_selected_config, on=['FileName', 'efConstruction', 'M'], how='right')
            current_train_df = pd.concat([current_train_df, selected_data_df], axis=0)

            current_selected_df_feature_labeled = current_train_df[
                ['efConstruction', 'M', 'efSearch', 'SIZE', 'q_SIZE', 'DIM', 'LID', 'Sum_K_MinDist',
                 'Sum_K_MaxDist', 'Sum_K_StdDist', 'q_K_MinRatio', 'q_K_MeanRatio', 'q_K_MaxRatio',
                 'q_K_StdRatio']]

            current_detected_df_feature_labeled = current_train_df[['efConstruction', 'M', 'efSearch', 'SIZE', 'q_SIZE', 'DIM', 'LID', 'Sum_K_MinDist',
                                                            'Sum_K_MaxDist',  'Sum_K_StdDist', 'q_K_MinRatio', 'q_K_MeanRatio', 'q_K_MaxRatio', 'q_K_StdRatio']]

            print('-------------------obtain new training data and perform data processing------------------')
            df_train, df_test, df_valid = split_data(current_train_df)
            df_train = pd.concat([df_train, df_test], axis=0)

            df_train_f, df_train_p = read_data_new(df_train)
            df_valid_f, df_valid_p = read_data_new(df_valid)

            df_train_f = df_train_f.drop(['FileName'], axis=1)
            df_valid_f = df_valid_f.drop(['FileName'], axis=1)

            whole_df_f = pd.concat([df_train_f, df_valid_f], axis=0)  # 还是要用全部数据更新标准化参数
            whole_feature = df2np(whole_df_f)
            whole_feature[:, 0] = np.log10(whole_feature[:, 0])
            whole_feature[:, 2:5] = np.log10(whole_feature[:, 2:5])

            whole_feature = np2ts(whole_feature).to(device)
            feature_scaler.fit(whole_feature)
            feature_scaler.save_parameters(None, new_feature_standard_path)

            feature_train = feature_df2feature_tensor(df_train_f, device, feature_scaler)
            feature_valid = feature_df2feature_tensor(df_valid_f, device, feature_scaler)

            performance_train = performance_df2performance_tensor(df_train_p, device, performance_scaler)
            performance_valid = performance_df2performance_tensor(df_valid_p, device, performance_scaler)

            performance_valid_raw = performance_scaler.inverse_transform(performance_valid)
            performance_valid_raw[:, 1:] = torch.pow(10, performance_valid_raw[:, 1:])

            dataset = CustomDataset(feature_train, performance_train)
            dataloader = DataLoader(dataset, batch_size=args.dipredict_batch_size, shuffle=True)

            print('-------------------update the model------------------')
            model = Direct_Predict_MLP(dipredict_layer_sizes)
            model.to(device)

            optimizer = optim.Adam(model.parameters(), lr=args.dipredict_lr, weight_decay=args.weight_decay)
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)

            dipredict_train(dataloader, model, optimizer, scheduler, feature_valid, performance_valid,
                            performance_valid_raw,
                            performance_scaler, args, new_model_save_path, None, device)

            df_test_f, df_test_p = read_data_new(real_data_test_df)
            df_test_f = df_test_f.drop(['FileName'], axis=1)

            feature_test = feature_df2feature_tensor(df_test_f, device, feature_scaler)
            performance_test_raw = df2np(df_test_p)
            performance_test_raw = np2ts(performance_test_raw).to(device)

            model.eval()
            with torch.no_grad():
                predicted_performances = model(feature_test)
                predicted_performances = performance_scaler.inverse_transform(predicted_performances)
                predicted_performances[:, 1:] = torch.pow(10, predicted_performances[:, 1:])

                mean_errors, mean_errors_percent, _ = calculate_errors(performance_test_raw, predicted_performances)
                print(f'mean_error:{mean_errors}, mean_error_percent:{mean_errors_percent}')

                test_error_dic['rec_MAE'].append(mean_errors[0].item())
                test_error_dic['ct_dc_MAE'].append(mean_errors[1].item())
                test_error_dic['st_dc_MAE'].append(mean_errors[2].item())
                test_error_dic['rec_MAPE'].append(mean_errors_percent[0].item())
                test_error_dic['ct_dc_MAPE'].append(mean_errors_percent[1].item())
                test_error_dic['st_dc_MAPE'].append(mean_errors_percent[2].item())

            print('-------------------model update completed, re-checking-------------------')
            t1 = time.time()
            detected_labeled_feature = feature_df2feature_tensor(current_detected_df_feature_labeled, device,
                                                                 feature_scaler)
            detected_labeled_feature_vectors = get_feature_vectors(model, detected_labeled_feature)

            selected_labeled_feature = feature_df2feature_tensor(current_selected_df_feature_labeled, device,
                                                                 feature_scaler)
            selected_labeled_feature_vectors = get_feature_vectors(model, selected_labeled_feature)

            min_distance = get_nn_dist(detected_labeled_feature_vectors, detected_labeled_feature_vectors, 2)
            distance_threshold = np.percentile(min_distance, 95)

            df_feature_unlabeled = get_input_feature(df_data_feature_test, current_df_config_unlabeled)

            df_feature_query = df_feature_unlabeled[
                ['efConstruction', 'M', 'efSearch', 'SIZE', 'q_SIZE', 'DIM', 'LID', 'Sum_K_MinDist',
                 'Sum_K_MaxDist', 'Sum_K_StdDist', 'q_K_MinRatio', 'q_K_MeanRatio', 'q_K_MaxRatio',
                 'q_K_StdRatio']]

            query_feature = feature_df2feature_tensor(df_feature_query, device, feature_scaler)
            query_feature_vectors = get_feature_vectors(model, query_feature)

            min_distance = get_nn_dist(detected_labeled_feature_vectors, query_feature_vectors, 1)
            mean_distance = np.mean(min_distance)
            dist_flag = (mean_distance <= distance_threshold)

            test_error_dic['distance_threshold'].append(distance_threshold)
            test_error_dic['mean_distance'].append(mean_distance)

            td = time.time()
            duration_time = td - ts
            detect_time = td - t1

            test_error_dic['duration_time'].append(duration_time)
            test_error_dic['each_detect_time'].append(detect_time)

            test_error_df = pd.DataFrame(test_error_dic)
            test_error_df['FileName'] = filename
            test_error_df.to_csv(test_error_path, index=False, mode='w', header=True)

            if dist_flag:
                break

    current_train_df.to_csv(new_train_data_path, mode='w', index=False)








































