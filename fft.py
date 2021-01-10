# -*- coding: utf-8 -*-
# @Author: xnchen
# @Email: xnchen97@gmail.com
# @Time: 2021/1/5
# @File: fft.py
import numpy as np
import matplotlib.pyplot as plt
import random
from typing import List, Optional, Union
from math import sqrt
from src import complex_convert

final_output_clip = [7, 7]


def fft(x: List[complex]):
    """
    DIT FFT for any length
    :param x:
    :return:
    """
    N = len(x)
    X = list()
    for k in range(0, N):
        X.append(x[k])

    fft_rec(X)
    return X


def fft_rec(X):
    N = len(X)

    if N <= 1:
        return

    even = np.array(X[0:N:2])
    odd = np.array(X[1:N:2])

    fft_rec(even)
    fft_rec(odd)

    for k in range(0, N // 2):
        t = np.exp(np.complex(0, -2 * np.pi * k / N)) * odd[k]
        X[k] = even[k] + t
        X[N // 2 + k] = even[k] - t


def DIF_FFT(x: List[np.complex],
            int_bit: Union[list, int, None] = None,
            frac_bit: Union[list, int, None] = None
            ) -> np.ndarray:
    assert len(x) == 64
    if not int_bit:
        int_bit = [10 for _ in range(7)]
    elif isinstance(int_bit, int):
        int_bit = [int_bit for _ in range(7)]
    if not frac_bit:
        frac_bit = [20 for _ in range(7)]
    elif isinstance(frac_bit, int):
        frac_bit = [frac_bit for _ in range(7)]

    # 第一级8bit fft
    x1 = np.zeros([8, 8], dtype=np.complex)  # n1, k2
    for i in range(8):
        x1[i] = DIF_FFT_8(x[i:64:8], int_bit[0:3], frac_bit[0:3])

    # 旋转矩阵
    revert_matrix(x1)
    x1 = complex_convert(x1, int_bit[3], frac_bit[3])

    # 第二级8bit fft
    x2 = np.zeros([8, 8], dtype=np.complex)
    for i in range(8):
        x2[i] = DIF_FFT_8(x1[:, i], int_bit[4:], frac_bit[4:])

    # 输出flatten
    res = np.zeros([64], dtype=complex)
    for i in range(8):
        for j in range(8):
            res[j * 8 + i] = x2[i, j]
    res = complex_convert(res, final_output_clip[0], final_output_clip[1])
    return res


def revert_matrix(m: np.ndarray):
    for n1 in range(8):
        for k2 in range(8):
            W = np.exp(np.complex(0, -2 * np.pi * (n1 * k2) / 64))
            m[n1, k2] *= W


def DIF_FFT_8(x: List[np.complex], int_bit: List, frac_bit: List) -> np.ndarray:
    assert len(x) == 8
    assert len(int_bit) >= 3, len(frac_bit) >= 3
    fft_res = np.array(x)
    DIF_stage1(fft_res)
    #print("stage1", fft_res)
    fft_res = complex_convert(fft_res, int_bit[0], frac_bit[0])
    DIF_stage2(fft_res)
    #print("stage2", fft_res)
    fft_res = complex_convert(fft_res, int_bit[1], frac_bit[1])
    DIF_stage3(fft_res)
    #print("stage3", fft_res)
    fft_res = complex_convert(fft_res, int_bit[2], frac_bit[2])

    return resort(fft_res)


def DIF_stage1(x: List[np.complex]):
    assert len(x) == 8

    W = [1,
         np.complex(sqrt(2) / 2, -sqrt(2) / 2),
         np.complex(0, -1),
         np.complex(-sqrt(2) / 2, -sqrt(2) / 2)]

    for i in range(4):
        x[i], x[i + 4] = \
            x[i] + x[i + 4], (x[i] - x[i + 4]) * W[i]


def DIF_stage2(x: List[np.complex]):
    assert len(x) == 8

    W = [1,
         np.complex(0, -1)]
    for i in range(2):
        for j in range(2):
            x[i * 4 + j], x[i * 4 + j + 2] = \
                x[i * 4 + j] + x[i * 4 + j + 2], (x[i * 4 + j] - x[i * 4 + j + 2]) * W[j]


def DIF_stage3(x: List[np.complex]):
    assert len(x) == 8

    for i in range(4):
        x[i * 2], x[i * 2 + 1] = \
            x[i * 2] + x[i * 2 + 1], x[i * 2] - x[i * 2 + 1]


def resort(x: np.ndarray):
    assert len(x) >= 8

    return x[[0, 4, 2, 6, 1, 5, 3, 7]]
