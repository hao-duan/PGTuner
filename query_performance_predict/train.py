import os
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.backends import cudnn
import random

from models import Direct_Predict_MLP
from utils import read_data_new, get_dataset, df2np, np2ts, CustomDataset, Scaler_minmax_new_gpu,calculate_errors
from trainer import dipredict_train
from Args import args


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

    os.makedirs(os.path.join(current_directory, 'scaler_paras'), exist_ok=True)
    os.makedirs(os.path.join(current_directory, 'model_checkpoints'), exist_ok=True)

    data_path = os.path.join(parent_directory, 'Data/train_data.csv')
    feature_standard_path = os.path.join(current_directory, 'scaler_paras/feature_standard.npz')

    print('-------------------load data-------------------')
    df_train, df_valid, df_test = get_dataset(data_path)

    df_train_f, df_train_p = read_data_new(df_train)
    df_valid_f, df_valid_p = read_data_new(df_valid)
    df_test_f, df_test_p = read_data_new(df_test)

    df_train_f = df_train_f.drop(['FileName'], axis=1)
    df_valid_f = df_valid_f.drop(['FileName'], axis=1)
    df_test_f = df_test_f.drop(['FileName'], axis=1)

    feature_train_raw = df2np(df_train_f)
    performance_train_raw = df2np(df_train_p)

    feature_valid_raw = df2np(df_valid_f)
    performance_valid_raw = df2np(df_valid_p)

    feature_test_raw = df2np(df_test_f)
    performance_test_raw = df2np(df_test_p)

    feature_train_raw[:, 0] = np.log10(feature_train_raw[:, 0])
    feature_train_raw[:, 2:5] = np.log10(feature_train_raw[:, 2:5])

    feature_valid_raw[:, 0] = np.log10(feature_valid_raw[:, 0])
    feature_valid_raw[:, 2:5] = np.log10(feature_valid_raw[:, 2:5])

    feature_test_raw[:, 0] = np.log10(feature_test_raw[:, 0])
    feature_test_raw[:, 2:5] = np.log10(feature_test_raw[:, 2:5])

    performance_train_raw[:, 1:] = np.log10(performance_train_raw[:, 1:])
    performance_valid_raw[:, 1:] = np.log10(performance_valid_raw[:, 1:])

    feature_train_raw = np2ts(feature_train_raw).to(device)
    feature_valid_raw = np2ts(feature_valid_raw).to(device)
    feature_test_raw = np2ts(feature_test_raw).to(device)
    performance_train_raw = np2ts(performance_train_raw).to(device)
    performance_valid_raw = np2ts(performance_valid_raw).to(device)
    performance_test_raw = np2ts(performance_test_raw).to(device)

    print('-------------------data normalization-------------------')
    feature_scaler = Scaler_minmax_new_gpu(6, device)
    if os.path.exists(feature_standard_path):
        feature_scaler.load_parameters(None, feature_standard_path, device)
    else:
        feature_raw = torch.cat((feature_train_raw, feature_valid_raw, feature_test_raw), dim=0)

        feature_scaler.fit(feature_raw)
        feature_scaler.save_parameters(None, feature_standard_path)

    feature_train = feature_scaler.transform(feature_train_raw)
    feature_valid = feature_scaler.transform(feature_valid_raw)
    feature_test = feature_scaler.transform(feature_test_raw)

    performance_scaler = Scaler_minmax_new_gpu(0, device)

    performance_train = performance_scaler.transform(performance_train_raw)
    performance_valid = performance_scaler.transform(performance_valid_raw)

    performance_valid_raw[:, 1:] = torch.pow(10, performance_valid_raw[:, 1:])

    dipredict_model_save_path = os.path.join(current_directory, 'model_checkpoints/{}_{}_{}_{}_checkpoint.pth'.format(
                                             args.dipredict_layer_sizes, args.dipredict_n_epochs, args.dipredict_batch_size, args.dipredict_lr))

    # writer = SummaryWriter(dipredict_loss_result_path)

    print('-------------------create Dataset and DataLoader-------------------')
    dataset = CustomDataset(feature_train, performance_train)
    dataloader = DataLoader(dataset, batch_size=args.dipredict_batch_size, shuffle=True)

    print('-------------------initialize the model and optimizer-------------------')
    layer_sizes = eval(args.dipredict_layer_sizes)

    model = Direct_Predict_MLP(layer_sizes)
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.dipredict_lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1000, gamma=0.1)

    print('-------------------start training-------------------')
    dipredict_train(dataloader, model, optimizer, scheduler, feature_valid, performance_valid, performance_valid_raw,
                    performance_scaler, args, dipredict_model_save_path, None, device)

    # writer.close()
    print('-------------------training completed, the prediction errors are:-------------------')
    predicted_performances = model(feature_test)
    predicted_performances = performance_scaler.inverse_transform(predicted_performances)
    predicted_performances[:, 1:] = torch.pow(10, predicted_performances[:, 1:])

    mean_errors, mean_errors_percent, mean_qerrors = calculate_errors(performance_test_raw, predicted_performances)
    print(f'mean_error:{mean_errors}, mean_error_percent:{mean_errors_percent}, mean_qerror:{mean_qerrors}')





