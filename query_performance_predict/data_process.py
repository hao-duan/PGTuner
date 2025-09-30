import os
import pandas as pd

def calculate_average(row):
    data_size_dic = {'glove_1_1.183514_100': [1183514, 1e4], 'paper_1_2.029997_200': [2029997, 1e4], 'sift_2_5_128_1': [5e7, 1e4],
                     'crawl_1_1.989995_300': [1989995, 1e4], 'nytimes_0_2.9_256': [290000, 1e3], 'tiny_1_1_384': [1e6, 1e3],
                     'msong_0_9.92272_420': [992272, 200], 'gist_1_1.0_960': [1e6, 1e3], 'deep_2_1_96': [1e7, 1e4], 'deep_1_1_96_1': [1e6, 1e4],
                     'sift_1_1_128_1': [1e6, 1e4], 'sift_1_2_128_1': [2e6, 1e4], 'sift_1_3_128_1': [3e6, 1e4], 'sift_1_4_128_1': [1e6, 4e4], 'sift_1_5_128_1': [5e6, 1e4],
                     'gist_1_1.0_960_25': [1e6, 1e3], 'gist_1_1.0_960_50': [1e6, 1e3], 'gist_1_1.0_960_75': [1e6, 1e3], 'gist_1_1.0_960_100': [1e6, 1e3]}

    for key, value in data_size_dic.items():
        if key in row['FileName']:
            row['average_construct_dc_counts'] = int(row['construct_dc_counts'] / value[0] + 1)
            row['average_search_dc_counts'] = int(row['search_dc_counts'] / value[1] + 1)
            row['whole_search_time'] = row['search_time'] * value[1]
            break
    return row

def get_data_feature(LID_feature_path, K_neighbor_feature_path, query_K_neighbor_feature_path, data_feature_path):
    df1 = pd.read_csv(LID_feature_path, sep=',', header=0)
    df2 = pd.read_csv(K_neighbor_feature_path, sep=',', header=0)
    df3 = pd.read_csv(query_K_neighbor_feature_path, sep=',', header=0)

    df1.drop(columns=['LIDTime'], inplace = True)
    df2.drop(columns=['SearchTime'], inplace=True)
    df3.drop(columns=['q_SearchTime'], inplace=True)

    result_df = pd.merge(df1, df2, on='FileName', how='left')
    result_df = pd.merge(result_df, df3, on='FileName', how='left')
    result_df.to_csv(data_feature_path, mode='w', index=False)

def get_train_test_data(data_feature_path, raw_data_path, data_path):
    df1 = pd.read_csv(data_feature_path, sep=',', header=0)
    df2 = pd.read_csv(raw_data_path, sep=',', header=0)

    df2 = df2.apply(calculate_average, axis=1)

    result_df = pd.merge(df1, df2, on='FileName', how='right')
    result_df.to_csv(data_path, mode='w', index=False)

def get_unlabeled_data(unlabeled_data_path):
    '''
    Used to generate unlabeled data in active learning algorithms.
    '''
    efCs = [20, 40, 60, 80, 100, 140, 180, 220, 260, 300, 340, 380, 420, 460, 500, 560, 620, 680, 740, 800]
    ms = [4, 8, 16, 24, 32, 48, 64, 80, 100]
    #ms = [4, 8, 12, 16, 20, 24, 32, 40, 48, 56, 64, 80, 100] # for sift50M dataset

    para_list = []

    for efC in efCs:
        for m in ms:
            if m <= efC:
                para_list.append([efC, m])
            else:
                break

    df = pd.DataFrame(para_list, columns=['efConstruction', 'M'])
    df.to_csv(unlabeled_data_path, mode='w', index=False)

if __name__ == '__main__':
    current_directory = os.getcwd()
    parent_directory = os.path.dirname(current_directory)

    LID_feature_path = os.path.join(parent_directory, 'Data/LID_data_feature.csv')
    K_neighbor_feature_path = os.path.join(parent_directory, 'Data/K_neighbor_dist_feature.csv')
    query_K_neighbor_feature_path = os.path.join(parent_directory, 'Data/query_K_neighbor_dist_ratio_feature.csv')
    data_feature_path = os.path.join(parent_directory, 'Data/data_feature.csv')

    train_data_performance_path = os.path.join(parent_directory, 'Data/index_performance_train.csv')
    test_data_performance_path = os.path.join(parent_directory, 'Data/index_performance_test_main.csv')

    train_data_path = os.path.join(parent_directory, 'Data/train_data.csv') 
    test_data_main_path = os.path.join(parent_directory, 'Data/test_data_main.csv')

    config_unit_data_path = os.path.join(parent_directory, 'Data/config_unit_data.csv')

    if not os.path.exists(data_feature_path):
        get_data_feature(LID_feature_path, K_neighbor_feature_path, query_K_neighbor_feature_path, data_feature_path)

    if not os.path.exists(train_data_path):
        get_train_test_data(data_feature_path, train_data_performance_path, train_data_path)

    if not os.path.exists(test_data_main_path):
        get_train_test_data(data_feature_path, test_data_performance_path, test_data_main_path)

    if not os.path.exists(config_unit_data_path):
        get_unlabeled_data(config_unit_data_path)





    