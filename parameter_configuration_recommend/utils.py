import torch
import numpy as np
import logging
import time
import datetime

class Scaler_para():
    def __init__(self):
        self.min = np.array([np.log10(20), 4, 1])
        self.max = np.array([np.log10(800), 100, np.log10(5000)])

    def transform(self, data):
        normalized_data = (data - self.min) / (self.max - self.min)
        return normalized_data

    def inverse_transform(self, data):
        raw_data = data * (self.max - self.min) + self.min
        return raw_data

class Scaler_para_nsg():
    def __init__(self):
        self.min = np.array([100, 100, 150, 5, 300, 1])
        self.max = np.array([400, 400, 350, 90, 600, np.log10(1500)])

    def transform(self, data):
        normalized_data = (data - self.min) / (self.max - self.min)
        return normalized_data

    def inverse_transform(self, data):
        raw_data = data * (self.max - self.min) + self.min
        return raw_data

class Scaler_para_gpu():
    def __init__(self, device):
        self.min = torch.tensor([torch.log10(torch.tensor(20)), 4, 1], dtype=torch.float32).to(device)
        self.max = torch.tensor([torch.log10(torch.tensor(800)), 100, torch.log10(torch.tensor(5000))], dtype=torch.float32).to(device)

    def transform(self, data):
        normalized_data = (data - self.min) / (self.max - self.min)
        return normalized_data

    def inverse_transform(self, data):
        raw_data = data * (self.max - self.min) + self.min
        return raw_data

class Scaler_para_gpu_nsg():
    def __init__(self, device):
        self.min = torch.tensor([100, 100, 150, 5, 300, 1], dtype=torch.float32).to(device)
        self.max = torch.tensor([400, 400, 350, 90, 600, torch.log10(torch.tensor(1500))], dtype=torch.float32).to(device)

    def transform(self, data):
        normalized_data = (data - self.min) / (self.max - self.min)
        return normalized_data

    def inverse_transform(self, data):
        raw_data = data * (self.max - self.min) + self.min
        return raw_data

class Scaler_state():
    def __init__(self, num):  # num = 12
        self.num = num

        self.min = np.array([np.log10(20), 4, 1, np.log10(20), 4, 1, 0, 2, 0, 0, 0, -1000])
        self.max = np.array([np.log10(800), 100, np.log10(5000), np.log10(800), 100, np.log10(5000), 1, np.log10(500000), 1, 1, 1, 1])

    def transform(self, data):
        normalized_data = (data - self.min) / (self.max - self.min)
        return normalized_data

    def inverse_transform(self, data):
        raw_data = data * (self.max - self.min) + self.min

        return raw_data

class Scaler_state_nsg():
    def __init__(self, num):  # num = 18
        self.num = num

        self.min = np.array([100, 100, 150, 5, 300, 1, 100, 100, 150, 5, 300, 1, 0, 1, 0, 0, 0, -1000])
        self.max = np.array([400, 400, 350, 90, 600, np.log10(1500), 400, 400, 350, 90, 600, np.log10(1500), 1, np.log10(50000), 1, 1, 1, 1])

    def transform(self, data):
        normalized_data = (data - self.min) / (self.max - self.min)
        return normalized_data

    def inverse_transform(self, data):
        raw_data = data * (self.max - self.min) + self.min

        return raw_data

class Logger:
    def __init__(self, name, log_file=''):
        self.log_file = log_file
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        sh = logging.StreamHandler()
        self.logger.addHandler(sh)
        if len(log_file) > 0:
            self.log2file = True
        else:
            self.log2file = False

    def _write_file(self, msg):
        if self.log2file:
            with open(self.log_file, 'a+') as f:
                f.write(msg + '\n')

    def get_timestr(self):
        timestamp = get_timestamp()
        date_str = time_to_str(timestamp)
        return date_str

    def warn(self, msg):
        msg = "%s[WARN] %s" % (self.get_timestr(), msg)
        self.logger.warning(msg)
        self._write_file(msg)

    def info(self, msg):
        msg = "%s[INFO] %s" % (self.get_timestr(), msg)
        self._write_file(msg)

    def error(self, msg):
        msg = "%s[ERROR] %s" % (self.get_timestr(), msg)
        self.logger.error(msg)
        self._write_file(msg)

def time_start():
    return time.time()

def time_end(start):
    end = time.time()
    delay = end - start
    return delay

def get_timestamp():
    return int(time.time())

def time_to_str(timestamp):
    return datetime.datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")