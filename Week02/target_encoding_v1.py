# coding = 'utf-8'
import numpy as np
import time
import functools
from target_encoding_v1_cy import target_mean_v3_cython
from target_encoding_v1_cy import target_mean_v4_cython
from target_encoding_v1_cy import target_mean_v5_cython


def log_execution_time(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        res = func(*args, **kwargs)
        end = time.perf_counter()
        print('{} took {} ms'.format(func.__name__, (end - start) * 1000))
        return res
    return wrapper

@log_execution_time
def target_mean_v1(data, y_name, x_name):
    result = np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        groupby_result = data[data.index != i].groupby([x_name], as_index=False).agg(['mean', 'count'])
        result[i] = groupby_result.loc[groupby_result.index == data.loc[i, x_name], (y_name, 'mean')]
    return result

@log_execution_time
def target_mean_v2(data, y_name, x_name):
    result = np.zeros(data.shape[0])
    value_dict = dict()
    count_dict = dict()
    for i in range(data.shape[0]):
        if data.loc[i, x_name] not in value_dict.keys():
            value_dict[data.loc[i, x_name]] = data.loc[i, y_name]
            count_dict[data.loc[i, x_name]] = 1
        else:
            value_dict[data.loc[i, x_name]] += data.loc[i, y_name]
            count_dict[data.loc[i, x_name]] += 1
    for i in range(data.shape[0]):
        result[i] = (value_dict[data.loc[i, x_name]] - data.loc[i, y_name]) / (count_dict[data.loc[i, x_name]] - 1)
    return result

@log_execution_time
def target_mean_v3(data, y_name, x_name):
    xs = data[x_name].values
    ys = data[y_name].values
    shape = data.shape[0]
    result = np.zeros(shape)
    return target_mean_v3_cython(xs, ys, shape, result, y_name, x_name)

@log_execution_time
def target_mean_v4(data, y_name, x_name):
    return target_mean_v4_cython(data, y_name, x_name)

@log_execution_time
def target_mean_v5(data, y_name, x_name):
    return target_mean_v5_cython(data, y_name, x_name)