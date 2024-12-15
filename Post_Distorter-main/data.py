import torch
from torch.utils.data import Dataset, DataLoader
import scipy.signal as signal 
import numpy as np
from led_model import led_model
class LED_Distortion(Dataset):
  def __init__(self, name, led_params, data_num=10000, bias=400, seed=27):
    self.name = name
    self.bias = bias
    self.seed = seed
    self.data_num = data_num + bias
    self.led_data = led_model(led_params, self.data_num)
  def __getitem__(self, index):
    # rng = np.random.RandomState(self.seed*index)
    # idx = rng.randint(self.bias, self.data_num - 1)
    bef_val = torch.Tensor([self.led_data['input'][index+self.bias]])
    cur_val = torch.Tensor([self.led_data['input'][index+self.bias]])
    # the label should be the  last num
    label = torch.Tensor([self.led_data['output'][index + self.bias - 1]])
    return bef_val, cur_val, label
  def __len__(self):
    return (self.data_num - self.bias)  