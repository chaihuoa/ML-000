# coding = 'utf-8'
# distutils: language=c++
import numpy as np
cimport numpy as np
import pandas as pd
import time
import functools
import cython
cimport cython
from libcpp.unordered_map cimport unordered_map


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

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef target_mean_v4_cython(data, str y_name, str x_name):
    cdef np.ndarray[long] xs = data[x_name].values
    cdef np.ndarray[long] ys = data[y_name].values
    cdef int n = data.shape[0]
    cdef np.ndarray[double] result = np.zeros(n)

    cdef value_dict = dict()
    cdef count_dict = dict()
    cdef int i = 0

    for i from 0<=i<n:
        index = xs[i]
        if index not in value_dict.keys():
            value_dict[index] = ys[i]
            count_dict[index] = 0
        else:
            value_dict[index] += ys[i]
            count_dict[index] += 1

    for i from 0<=i<n:
        index = xs[i]
        result[i] = (value_dict[index] - ys[i]) / count_dict[index]

    return result

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef target_mean_v5_cython(data, str y_name, str x_name):
    cdef np.ndarray[long] xs = data[x_name].values
    cdef np.ndarray[long] ys = data[y_name].values
    cdef int n = data.shape[0]
    cdef np.ndarray[double] result = np.zeros(n)

    cdef unordered_map[long, long] value_dict
    cdef unordered_map[long, long] count_dict

    for i in range(n):
        x_value, y_value = xs[i], ys[i]
        if not value_dict.count(x_value):
            value_dict[x_value] = y_value
            count_dict[x_value] = 1
        else:
            value_dict[x_value] += y_value
            count_dict[x_value] += 1
    for i in range(n):
        x_value, y_value = xs[i], ys[i]
        if count_dict[x_value] == 1:
            result[i] = 0.0
        else:
            result[i] = (value_dict[x_value] - y_value) / (count_dict[x_value] - 1)

    return result