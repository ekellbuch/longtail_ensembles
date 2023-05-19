import pytorch_lightning as pl
from torch import nn

from torchmetrics import Accuracy
import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np
from scipy import stats

def add_dicts(dict1, dict2):
  """
  Add up the values of two dictionaries.

  Args:
      dict1 (dict): First dictionary.
      dict2 (dict): Second dictionary.

  Returns:
      dict: A new dictionary with summed up values.
  """
  result_dict = dict1.copy()

  for key, value in dict2.items():
    if key in result_dict:
      result_dict[key] += value
    else:
      result_dict[key] = value

  return result_dict




def read_dataset(datafile):
  # % ret het binned ensemble
  data3 = pd.read_csv(datafile, index_col=False, header=0)
  # filter het
  #data3 = data3[data3['type']=='het']
  return data3

def z_score_difference(beta1, beta2, se_1, se_2):
  z = (beta1 - beta2) / np.sqrt(se_1 ** 2 + se_2 ** 2)
  p_value = stats.norm.sf(np.abs(z))
  return z, p_value


def get_maxxy(data, datas):
  max_y_value = np.ceil(np.max([data[:, 0].max(), datas[:, 0].max()]))
  max_x_value = np.ceil(np.max([data[:, 1].max(), datas[:, 1].max()]))
  max_x_value = np.max([max_y_value / 2.5, max_x_value])
  return max_x_value, max_y_value


# fit all these points using linear regression


def linear_fit(x, y):
  if x.ndim == 1:
    X = x[:, None]
  lm = LinearRegression()
  lm.fit(X, y)
  params = np.append(lm.intercept_, lm.coef_)
  predictions = lm.predict(X)
  newX = np.append(np.ones((len(X), 1)), X, axis=1)
  MSE = (sum((y - predictions) ** 2)) / (len(newX) - len(newX[0]))
  var_b = MSE * (np.linalg.inv(np.dot(newX.T, newX)).diagonal())
  sd_b = np.sqrt(var_b)
  ts_b = params / sd_b
  # print(ts_b)
  p_values = [2 * (1 - stats.t.cdf(np.abs(i), (len(newX) - len(newX[0])))) for i in ts_b]
  # print(p_values)
  return predictions, params, sd_b, ts_b, p_values


pretty_data_names = {
  "cifar10": "CIFAR10",
  "cifar10_1": "CIFAR10.1",
  "cinic10": "CINIC10",
  "imagenet": "ImageNet",
  "imagenetv2mf": "ImageNetV2 MF",
  "imagenet_c_fog_1": "ImageNet-C \n Fog 1",
  "imagenet_c_fog_3": "ImageNet-C \n Fog 3",
  "imagenet_c_fog_5": "ImageNet-C \n Fog 5",
  "imagenet_c_gaussian_noise_1": "ImageNet-C \n Gaussian 1",
  "imagenet_c_gaussian_noise_3": "ImageNet-C \n Gaussian 3",
  "imagenet_c_gaussian_noise_5": "ImageNet-C \n Gaussian 5",

}