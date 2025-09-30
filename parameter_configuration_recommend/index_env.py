import os
import torch
import torch.optim as optim
import numpy as np
import pickle
from query_performance_predict.models import Direct_Predict_MLP, Direct_Predict_MLP_nsg
from query_performance_predict.utils import np2ts, Scaler_minmax_new_gpu, Scaler_minmax_new_gpu_nsg, load_model
from utils import Scaler_para, Scaler_para_nsg, Scaler_state, Scaler_state_nsg

'''
The environment of reinforcement learning for PG configuration tuning.
'''

class IndexEnv(object): 
    def __init__(self, num_dataset, default_performance, target_rec_lis, args_r, args_p, predict_model_save_path, standard_path, dv):
        self.device = dv

        self.args_p = args_p
        self.predict_model = Direct_Predict_MLP(eval(args_p.dipredict_layer_sizes)).to(self.device)
        self._get_predict_model(predict_model_save_path)  # load QPP model

        self.feature_scaler = Scaler_minmax_new_gpu(6, dv)
        self.performance_scaler = Scaler_minmax_new_gpu(0, dv)
        self.para_scaler = Scaler_para()  
        self.state_scaler = Scaler_state(12)
        self._load_scaler(standard_path)

        self.num_dataset = num_dataset
        self.args_r = args_r

        self.target_rec = np.tile(np.array(target_rec_lis).reshape(-1, 1), (num_dataset, 1))

        self.score = 0.0
        self.steps = 0
        self.max_steps = args_r.max_steps
        self.nochange_steps = 0

        self.default_paras = np.tile(np.array([[20, 4, 10]]), (self.target_rec.shape[0], 1))  # start tuning from the worst configuration
        self.default_index_performance = default_performance.copy()  
        self.best_index_performance = default_performance.copy()  # store the currently best performance

    def _get_predict_model(self, model_save_path):
        optimizer = optim.Adam(self.predict_model.parameters(), lr=self.args_p.dipredict_lr, weight_decay=self.args_p.weight_decay)
        self.predict_model, _, _ = load_model(self.predict_model, optimizer, model_save_path)
        self.predict_model.to(self.device)
        self.predict_model.eval()

    def _load_scaler(self, standard_path):
        self.feature_scaler.load_parameters(None, standard_path, self.device)

    def _get_action(self, actions): 
        paras = self.para_scaler.inverse_transform(actions)  
        paras[:, 0] = np.power(10, paras[:, 0])
        paras[:, 2] = np.power(10, paras[:, 2])
        paras = np.floor(paras + 0.5)  

        paras[:, 0] = np.where(paras[:, 0] < paras[:, 1], paras[:, 1], paras[:, 0])
        return paras

    def _get_index_performance(self, feature_input):
        feature_input = np2ts(feature_input).to(self.device)
        feature_input_scaled = self.feature_scaler.transform(feature_input)

        with torch.no_grad():
            index_performance = self.predict_model(feature_input_scaled)

        real_index_performance = self.performance_scaler.inverse_transform(index_performance)
        real_index_performance[:, 1:] = torch.pow(10, real_index_performance[:, 1:])

        real_index_performance = real_index_performance.cpu().numpy()
        return real_index_performance 

    @staticmethod
    def _get_performance_improvement(last_index_performance, current_index_performance):
        st_counts_dec = (last_index_performance[:, 2] - current_index_performance[:, 2]) / last_index_performance[:, 2]
        return st_counts_dec

    def _get_best_now(self, filename):
        with open(filename, 'rb') as f:
            best_index_performance = pickle.load(f)
        return best_index_performance

    def _get_best_paras_now(self, filename):
        with open(filename, 'rb') as f:
            best_paras = pickle.load(f)
        return best_paras

    def _record_best(self, cur_index_performance, cur_paras, performance_filename, paras_filename): 
        if os.path.exists(performance_filename): 
            best_index_performance = self._get_best_now(performance_filename)
            best_paras = self._get_best_paras_now(paras_filename)

            target_rec = self.target_rec.reshape(-1)
            best_rec = best_index_performance[:, 0]
            cur_rec = cur_index_performance[:, 0]

            target_dec = self._get_performance_improvement(best_index_performance, cur_index_performance)

            cond1 = (best_rec < target_rec)
            cond2 = (best_rec >= target_rec)
            cond3 = (cur_rec < target_rec)
            cond4 = (cur_rec >= target_rec)
            cond5 = (cur_rec > best_rec)
            cond6 = (target_dec > 0)

            cond_a = (cond1 & cond4)
            cond_b = (cond1 & cond3 & cond5)
            cond_c = (cond2 & cond4 & cond6)

            if cond_a.any() or cond_b.any() or cond_c.any():
                self.nochange_steps = 0

                best_index_performance[cond_a] = cur_index_performance[cond_a]
                best_paras[cond_a] = cur_paras[cond_a]

                best_index_performance[cond_b] = cur_index_performance[cond_b]
                best_paras[cond_b] = cur_paras[cond_b]

                best_index_performance[cond_c] = cur_index_performance[cond_c]
                best_paras[cond_c] = cur_paras[cond_c]

                with open(performance_filename, 'wb') as f:
                    pickle.dump(best_index_performance, f)

                with open(paras_filename, 'wb') as f:
                    pickle.dump(best_paras, f)
            else:
                self.nochange_steps += 1

            return cond_c
        else:
            with open(performance_filename, 'wb') as f:
                pickle.dump(cur_index_performance, f)

            with open(paras_filename, 'wb') as f:
                pickle.dump(cur_paras, f)

            return np.zeros(cur_index_performance.shape[0], dtype=bool)

    def _initialize(self):
        self.steps = 0
        self.score = 0.0
        self.nochange_steps = 0

        num = self.default_index_performance.shape[0]

        self.best_index_performance = self.default_index_performance.copy()

        target_rec = self.target_rec.reshape(-1)
        best_rec = self.best_index_performance[:, 0]

        deltat20 = (best_rec - target_rec).reshape((num, 1))
        delta0 = deltat20.copy()
        deltat = np.zeros((num, 1))

        target_dec = np.zeros((num, 1))

        cur_index_performance = self.default_index_performance.copy()
        
        cur_state_index_performance = np.zeros((num, 2))
        cur_state_index_performance[:, 0] = cur_index_performance[:, 0]
        cur_state_index_performance[:, 1] = cur_index_performance[:, 2]

        init_state = np.concatenate((np.copy(self.default_paras), np.copy(self.default_paras), cur_state_index_performance, delta0, deltat20, deltat, target_dec), axis=1) 

        init_state[:, 0] = np.log10(init_state[:, 0])
        init_state[:, 2:4] = np.log10(init_state[:, 2:4])
        init_state[:, 5] = np.log10(init_state[:, 5])
        init_state[:, 7] = np.log10(init_state[:, 7])

        init_state_ = self.state_scaler.transform(init_state)

        return init_state_

    def _get_next_state(self, cur_index_performance, best_index_performance, cur_paras, best_paras, target_rec, num):
        best_rec = best_index_performance[:, 0]
        cur_rec = cur_index_performance[:, 0]

        delta0 = (cur_rec - target_rec).reshape((num, 1))
        deltat20 = (best_rec - target_rec).reshape((num, 1))
        deltat = (cur_rec - best_rec).reshape((num, 1))

        target_dec = self._get_performance_improvement(best_index_performance, cur_index_performance)
        target_dec= target_dec.reshape((num, 1))

        cur_state_index_performance = np.zeros((num, 2))
        cur_state_index_performance[:, 0] = cur_index_performance[:, 0]
        cur_state_index_performance[:, 1] = cur_index_performance[:, 2]

        next_state = np.concatenate((cur_paras, best_paras, cur_state_index_performance, delta0, deltat20, deltat, target_dec), axis=1)
        return next_state

    def _step(self, actions, data_feature, performance_filename, paras_filename):
        self.steps += 1

        num = self.target_rec.shape[0]
        target_rec = self.target_rec.reshape(-1)

        cur_paras = self._get_action(actions)

        feature_input = np.concatenate((np.copy(cur_paras), np.copy(data_feature)), axis=1)
        feature_input[:, 0] = np.log10(feature_input[:, 0])
        feature_input[:, 2] = np.log10(feature_input[:, 2])

        cur_index_performance = self._get_index_performance(feature_input)

        reward, average_reward = self._get_reward(cur_index_performance)

        _ = self._record_best(cur_index_performance, cur_paras, performance_filename, paras_filename)

        best_now_performance = self._get_best_now(performance_filename)
        best_now_paras = self._get_best_paras_now(paras_filename)

        self.best_index_performance = best_now_performance.copy() 

        next_state = self._get_next_state(cur_index_performance, best_now_performance, cur_paras, best_now_paras, target_rec, num)

        next_state[:, 0] = np.log10(next_state[:, 0])
        next_state[:, 2:4] = np.log10(next_state[:, 2:4])
        next_state[:, 5] = np.log10(next_state[:, 5])
        next_state[:, 7] = np.log10(next_state[:, 7])

        next_state_ = self.state_scaler.transform(next_state)

        terminate = np.zeros((num, 1), dtype=bool)

        return reward, next_state_, terminate, self.score, average_reward, cur_index_performance, cur_paras

    @staticmethod
    def _calculate_reward(delta):
        reward_positive = (1 + delta) ** 2 - 1
        reward_negative = -(1 - delta) ** 2 + 1

        _reward = np.where(delta >= 0, reward_positive, reward_negative)
        return _reward

    @staticmethod
    def _calculate_reward_rec(delta0, deltat, deltat20):
        _reward = np.zeros_like(delta0)

        cond1 = (deltat20 < 0) & (delta0 < 0)
        cond2 = (deltat20 < 0) & (delta0 >= 0)
        cond3 = (deltat20 > 0) & (delta0 < 0)

        reward_cond1 = -(1 - delta0) ** 2 + 1
        reward_cond2 = ((1 + delta0) ** 2) * (1 + deltat)
        reward_cond3 = (-(1 - delta0) ** 2) * (1 - deltat)

        _reward[cond1] = reward_cond1[cond1]
        _reward[cond2] = reward_cond2[cond2]
        _reward[cond3] = reward_cond3[cond3]

        return _reward

    def _get_reward(self, cur_index_performance):
        target_rec = self.target_rec.reshape(-1)
        best_rec = self.best_index_performance[:, 0] 
        cur_rec = cur_index_performance[:, 0]

        delta0 = cur_rec - target_rec
        deltat = cur_rec - best_rec
        deltat20 = best_rec - target_rec

        reward = self._calculate_reward_rec(delta0, deltat, deltat20)

        st_counts_dec = self._get_performance_improvement(self.best_index_performance, cur_index_performance)
        target_dec = st_counts_dec

        counts_reward = self._calculate_reward(target_dec)

        condition = (best_rec >= target_rec) & (cur_rec >= target_rec)
        reward[condition] = counts_reward[condition]

        average_reward = np.mean(reward)
        self.score += average_reward

        reward = np.where(condition & (reward > 0), reward * self.args_r.pec_reward, reward)
        return reward, average_reward

class IndexEnv_nsg(object):
    def __init__(self, num_dataset, default_performance, target_rec_lis, args_r, args_p, predict_model_save_path,
                 standard_path, dv):
        self.device = dv

        self.args_p = args_p
        self.predict_model = Direct_Predict_MLP_nsg(eval(args_p.dipredict_layer_sizes_nsg)).to(self.device)
        self._get_predict_model(predict_model_save_path)

        self.feature_scaler = Scaler_minmax_new_gpu_nsg(9, dv)
        self.performance_scaler = Scaler_minmax_new_gpu_nsg(0, dv)
        self.para_scaler = Scaler_para_nsg()
        self.state_scaler = Scaler_state_nsg(18)
        self._load_scaler(standard_path)

        self.num_dataset = num_dataset
        self.args_r = args_r

        self.target_rec = np.tile(np.array(target_rec_lis).reshape(-1, 1), (num_dataset, 1))

        self.score = 0.0
        self.steps = 0
        self.max_steps = args_r.max_steps
        self.nochange_steps = 0

        self.default_paras = np.tile(np.array([[100, 100, 150, 5, 300, 10]]), (self.target_rec.shape[0], 1))
        self.default_index_performance = default_performance.copy()
        self.best_index_performance = default_performance.copy()

    def _get_predict_model(self, model_save_path):
        optimizer = optim.Adam(self.predict_model.parameters(), lr=self.args_p.dipredict_lr, weight_decay=self.args_p.weight_decay)
        self.predict_model, _, _ = load_model(self.predict_model, optimizer, model_save_path)
        self.predict_model.to(self.device)
        self.predict_model.eval()

    def _load_scaler(self, standard_path):
        self.feature_scaler.load_parameters(None, standard_path, self.device)

    def _get_action(self, actions):
        paras = self.para_scaler.inverse_transform(actions)
        paras[:, 5] = np.power(10, paras[:, 5])
        paras = np.floor(paras + 0.5)

        paras[:, 1] = np.where(paras[:, 1] < paras[:, 0], paras[:, 0], paras[:, 1])
        return paras

    def _get_index_performance(self, feature_input):
        feature_input = np2ts(feature_input).to(self.device)
        feature_input_scaled = self.feature_scaler.transform(feature_input)

        with torch.no_grad():
            index_performance = self.predict_model(feature_input_scaled)

        real_index_performance = self.performance_scaler.inverse_transform(index_performance)
        real_index_performance[:, 1] = torch.pow(10, real_index_performance[:, 1])

        real_index_performance = real_index_performance.cpu().numpy()
        return real_index_performance

    @staticmethod
    def _get_performance_improvement(last_index_performance, current_index_performance):
        st_counts_dec = (last_index_performance[:, 1] - current_index_performance[:, 1]) / last_index_performance[:, 1]
        return st_counts_dec

    def _get_best_now(self, filename):
        with open(filename, 'rb') as f:
            best_index_performance = pickle.load(f)
        return best_index_performance

    def _get_best_paras_now(self, filename):
        with open(filename, 'rb') as f:
            best_paras = pickle.load(f)
        return best_paras

    def _record_best(self, cur_index_performance, cur_paras, performance_filename, paras_filename):  
        if os.path.exists(performance_filename):  
            best_index_performance = self._get_best_now(performance_filename)
            best_paras = self._get_best_paras_now(paras_filename)

            target_rec = self.target_rec.reshape(-1)
            best_rec = best_index_performance[:, 0]
            cur_rec = cur_index_performance[:, 0]

            target_dec = self._get_performance_improvement(best_index_performance, cur_index_performance)

            cond1 = (best_rec < target_rec)
            cond2 = (best_rec >= target_rec)
            cond3 = (cur_rec < target_rec)
            cond4 = (cur_rec >= target_rec)
            cond5 = (cur_rec > best_rec)
            cond6 = (target_dec > 0)

            cond_a = (cond1 & cond4)
            cond_b = (cond1 & cond3 & cond5)
            cond_c = (cond2 & cond4 & cond6)

            if cond_a.any() or cond_b.any() or cond_c.any():
                self.nochange_steps = 0

                best_index_performance[cond_a] = cur_index_performance[cond_a]
                best_paras[cond_a] = cur_paras[cond_a]

                best_index_performance[cond_b] = cur_index_performance[cond_b]
                best_paras[cond_b] = cur_paras[cond_b]

                best_index_performance[cond_c] = cur_index_performance[cond_c]
                best_paras[cond_c] = cur_paras[cond_c]

                with open(performance_filename, 'wb') as f:
                    pickle.dump(best_index_performance, f)

                with open(paras_filename, 'wb') as f:
                    pickle.dump(best_paras, f)
            else:
                self.nochange_steps += 1

            return cond_c
        else:
            with open(performance_filename, 'wb') as f:
                pickle.dump(cur_index_performance, f)

            with open(paras_filename, 'wb') as f:
                pickle.dump(cur_paras, f)

            return np.zeros(cur_index_performance.shape[0], dtype=bool)

    def _initialize(self):
        self.steps = 0
        self.score = 0.0
        self.nochange_steps = 0

        num = self.default_index_performance.shape[0]

        self.best_index_performance = self.default_index_performance.copy()

        target_rec = self.target_rec.reshape(-1)
        best_rec = self.best_index_performance[:, 0]

        deltat20 = (best_rec - target_rec).reshape((num, 1))
        delta0 = deltat20.copy()
        deltat = np.zeros((num, 1))

        target_dec = np.zeros((num, 1))

        cur_index_performance = self.default_index_performance.copy()

        cur_state_index_performance = np.zeros((num, 2))
        cur_state_index_performance[:, 0] = cur_index_performance[:, 0]
        cur_state_index_performance[:, 1] = cur_index_performance[:, 1]

        init_state = np.concatenate((np.copy(self.default_paras), np.copy(self.default_paras), cur_state_index_performance, delta0, deltat20, deltat, target_dec), axis=1)

        init_state[:, 5] = np.log10(init_state[:, 5])
        init_state[:, 11] = np.log10(init_state[:, 11])
        init_state[:, 13] = np.log10(init_state[:, 13])

        init_state_ = self.state_scaler.transform(init_state)  # 状态向量要归一化再存储

        return init_state_

    def _get_next_state(self, cur_index_performance, best_index_performance, cur_paras, best_paras, target_rec, num):
        best_rec = best_index_performance[:, 0]
        cur_rec = cur_index_performance[:, 0]

        delta0 = (cur_rec - target_rec).reshape((num, 1))
        deltat20 = (best_rec - target_rec).reshape((num, 1))
        deltat = (cur_rec - best_rec).reshape((num, 1))

        target_dec = self._get_performance_improvement(best_index_performance, cur_index_performance)
        target_dec = target_dec.reshape((num, 1))

        cur_state_index_performance = np.zeros((num, 2))
        cur_state_index_performance[:, 0] = cur_index_performance[:, 0]
        cur_state_index_performance[:, 1] = cur_index_performance[:, 1]

        next_state = np.concatenate((cur_paras, best_paras, cur_state_index_performance, delta0, deltat20, deltat, target_dec), axis=1)

        return next_state

    def _step(self, actions, data_feature, performance_filename, paras_filename):
        self.steps += 1

        num = self.target_rec.shape[0]
        target_rec = self.target_rec.reshape(-1)

        cur_paras = self._get_action(actions)

        feature_input = np.concatenate((np.copy(cur_paras), np.copy(data_feature)), axis=1)
        feature_input[:, 5] = np.log10(feature_input[:, 5])

        cur_index_performance = self._get_index_performance(feature_input)

        reward, average_reward = self._get_reward(cur_index_performance)

        _ = self._record_best(cur_index_performance, cur_paras, performance_filename, paras_filename)

        best_now_performance = self._get_best_now(performance_filename)
        best_now_paras = self._get_best_paras_now(paras_filename)

        self.best_index_performance = best_now_performance.copy()

        next_state = self._get_next_state(cur_index_performance, best_now_performance, cur_paras, best_now_paras, target_rec, num)

        next_state[:, 5] = np.log10(next_state[:, 5])
        next_state[:, 11] = np.log10(next_state[:, 11])
        next_state[:, 13] = np.log10(next_state[:, 13])

        next_state_ = self.state_scaler.transform(next_state)

        terminate = np.zeros((num, 1), dtype=bool)

        return reward, next_state_, terminate, self.score, average_reward, cur_index_performance, cur_paras

    @staticmethod
    def _calculate_reward(delta):
        reward_positive = (1 + delta) ** 2 - 1
        reward_negative = -(1 - delta) ** 2 + 1

        _reward = np.where(delta >= 0, reward_positive, reward_negative)
        return _reward

    @staticmethod
    def _calculate_reward_rec(delta0, deltat, deltat20):
        _reward = np.zeros_like(delta0)

        cond1 = (deltat20 < 0) & (delta0 < 0)
        cond2 = (deltat20 < 0) & (delta0 >= 0)
        cond3 = (deltat20 > 0) & (delta0 < 0)

        reward_cond1 = -(1 - delta0) ** 2 + 1
        reward_cond2 = ((1 + delta0) ** 2) * (1 + deltat)
        reward_cond3 = (-(1 - delta0) ** 2) * (1 - deltat)

        _reward[cond1] = reward_cond1[cond1]
        _reward[cond2] = reward_cond2[cond2]
        _reward[cond3] = reward_cond3[cond3]

        return _reward

    def _get_reward(self, cur_index_performance):
        target_rec = self.target_rec.reshape(-1)
        best_rec = self.best_index_performance[:, 0]
        cur_rec = cur_index_performance[:, 0]

        delta0 = cur_rec - target_rec
        deltat = cur_rec - best_rec
        deltat20 = best_rec - target_rec

        reward = self._calculate_reward_rec(delta0, deltat, deltat20)

        st_counts_dec = self._get_performance_improvement(self.best_index_performance, cur_index_performance)
        target_dec = st_counts_dec

        counts_reward = self._calculate_reward(target_dec)

        condition = (best_rec >= target_rec) & (cur_rec >= target_rec)
        reward[condition] = counts_reward[condition]

        average_reward = np.mean(reward)
        self.score += average_reward

        reward = np.where(condition & (reward > 0), reward * self.args_r.pec_reward, reward)

        return reward, average_reward