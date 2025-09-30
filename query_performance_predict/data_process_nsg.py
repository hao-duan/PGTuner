import os
import pandas as pd

def calculate_average(row):
    data_size_dic = {'paper_0_2_200_1': [2e5, 1e4], 'deep_0_1_96_1': [1e5, 1e4], 'gist_0_1_960_1': [1e5, 1e3],}

    for key, value in data_size_dic.items():
        if key in row['FileName']:
            row['average_NSG_s_dc_counts'] = int(row['NSG_s_dc_counts'] / value[1] + 1)
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

def get_unlabeled_data_NSG(unlabeled_data_NSG_path):
    '''
    Used to generate unlabeled data in active learning algorithm.
    '''
    L_nsg_Cs = [150, 200, 250, 300, 350]
    R_nsgs = [5, 10, 15, 20, 25, 30, 35, 40, 50, 70, 90]
    Cs = [300, 400, 500, 600]

    NSG_para_list = []

    for L_nsg_C in L_nsg_Cs:
        for R_nsg in R_nsgs:
            for C in Cs:
                if R_nsg <= C:
                    NSG_para_list.append((L_nsg_C, R_nsg, C))
                else:
                    break

    df = pd.DataFrame(NSG_para_list, columns=['L_nsg_C', 'R_nsg', 'C'])
    df.to_csv(unlabeled_data_NSG_path, mode='w', index=False)

def get_unlabeled_data_KNN(unlabeled_data_KNN_path):
    '''
    Used to generate unlabeled data in active learning algorithm.
    '''
    Ks = [100, 200, 300, 400]
    Ls = [100, 150, 200, 250, 300, 350, 400]
    KNN_para_list = []
    for L in Ls:
        for K in Ks:
            if K <= L:
                KNN_para_list.append((K, L))
            else:
                break

    df = pd.DataFrame(KNN_para_list, columns=['K', 'L'])
    df.to_csv(unlabeled_data_KNN_path, mode='w', index=False)

if __name__ == '__main__':
    current_directory = os.getcwd()
    parent_directory = os.path.dirname(current_directory)

    LID_feature_path = os.path.join(parent_directory, 'Data/LID_data_feature.csv')
    K_neighbor_feature_path = os.path.join(parent_directory, 'Data/K_neighbor_dist_feature.csv')
    query_K_neighbor_feature_path = os.path.join(parent_directory, 'Data/query_K_neighbor_dist_ratio_feature.csv')
    data_feature_path = os.path.join(parent_directory, 'NSG_KNNG/Data/data_feature.csv')

    train_data_performance_path = os.path.join(parent_directory, 'NSG_KNNG/Data/index_performance_train.csv')
    test_data_performance_path = os.path.join(parent_directory, 'NSG_KNNG/Data/index_performance_test_main.csv')

    train_data_path = os.path.join(parent_directory, 'NSG_KNNG/Data/train_data.csv')
    test_data_main_path = os.path.join(parent_directory, 'NSG_KNNG/Data/test_data_main.csv')

    KNN_config_unit_data_path = os.path.join(parent_directory, 'NSG_KNNG/Data/KNN_config_unit_data.csv')
    NSG_config_unit_data_path = os.path.join(parent_directory, 'NSG_KNNG/Data/NSG_config_unit_data.csv')

    # if not os.path.exists(data_feature_path):
    #     get_data_feature(LID_feature_path, K_neighbor_feature_path, query_K_neighbor_feature_path, data_feature_path)

    if not os.path.exists(train_data_path):
        get_train_test_data(data_feature_path, train_data_performance_path, train_data_path)

    if not os.path.exists(test_data_main_path):
        get_train_test_data(data_feature_path, test_data_performance_path, test_data_main_path)

    if not os.path.exists(KNN_config_unit_data_path):
        get_unlabeled_data_KNN(KNN_config_unit_data_path)

    if not os.path.exists(NSG_config_unit_data_path):
        get_unlabeled_data_NSG(NSG_config_unit_data_path)






    