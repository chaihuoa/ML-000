# coding = 'utf-8'
# distutils: language=c++
import numpy as np
cimport numpy as np
import pandas as pd
import time
import functools
import cython
cimport cython


cpdef target_mean_v3_cython(np.ndarray[long] xs, np.ndarray[long] ys, int shape, np.ndarray result, str y_name, str x_name):
    value_dict = dict()
    count_dict = dict()
    for i in range(shape):
        index = xs[i]
        if index not in value_dict.keys():
            value_dict[index] = ys[i]
            count_dict[index] = 0
        else:
            value_dict[index] += ys[i]
            count_dict[index] += 1
    for i in range(shape):
        index = xs[i]
        result[i] = (value_dict[index] - ys[i]) / count_dict[index]
    return result
