# -*- coding: utf-8 -*-
# @Author: xnchen
# @Email: xnchen97@gmail.com
# @Time: 2021/1/5
# @File: src.py

import numpy as np
from typing import Tuple


def SNR(signal: np.ndarray, gt: np.ndarray) -> float:
    signal = np.asanyarray(signal)
    gt = np.asanyarray(gt)
    error = np.sum(np.abs(signal - gt) ** 2)
    output = np.sum(np.abs(gt) ** 2)
    snr = 10 * np.log10(output / error)
    return snr


def convert_to_fix(arr: np.ndarray, bit: int) -> np.ndarray:
    """
    从Array[np.float]进行量化，量化大小为bit+1（有一位补码位）
    返回Array[int]格式的量化数字
    """
    arr1 = arr.copy().astype(np.float32)
    arr1[arr1 < 0] = 0.0
    arr1 = np.round(np.abs(arr1) * (2 ** bit))

    arr2 = arr.copy().astype(np.float32)
    arr2[arr2 > 0] = 0.0
    arr2 = -np.round(np.abs(-arr2) * 2 ** bit)

    res = arr1 + arr2
    return res.astype(np.int64)


def convert_to_float(arr: np.ndarray, bit: int) -> np.ndarray:
    """
    将 convert_to_fix的Array[int]格式的量化数字再转成Array[np.float]
    """
    return arr / (2 ** bit)


def fix_fractional_part(x: np.ndarray, bit: int) -> np.ndarray:
    """
    小数的量化
    """
    return convert_to_float(convert_to_fix(x, bit), bit)


def fix_integer_part(x: np.ndarray, bit: int) -> np.ndarray:
    """
    整数的量化
    """
    # print("\tx: ",x,
    #      "\n\tinteger_fix: ",np.clip(x, -2 ** bit + 1, 2 ** bit - 1))
    return np.clip(x, -2 ** bit+1 , 2 ** bit )


def real_convert(x: np.ndarray, frac_bit: int, int_bit: int) -> np.ndarray:
    """
    实数的量化：小数和整数部分是分开的
    """
    int_part, frac_part = divide_int_frac(x)
    int_part = fix_integer_part(int_part, int_bit)
    frac_part = fix_fractional_part(frac_part, frac_bit)
    return merge_int_frac(int_part, frac_part)


def merge_int_frac(int_part: np.ndarray, frac_part: np.ndarray) -> np.ndarray:
    return int_part + frac_part


def divide_int_frac(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    将array中的数字的整数和小数部分分开
    :param x:
    :return:
    """
    # 整数部分
    arr1 = x.copy().astype(np.float32)
    arr1[arr1 < 1] = 0.0
    arr1 = np.floor(arr1)
    arr2 = x.copy().astype(np.float32)
    arr2[arr2 > -1] = 0.0
    arr2 = np.ceil(arr2)
    int_part = arr1 + arr2

    # 浮点数部分
    arr3 = x.copy().astype(np.float32)
    frac_part = arr3 - int_part
    return int_part, frac_part


def complex_convert(x: np.ndarray,
                    int_bit: int,
                    frac_bit: int) -> np.ndarray:
    """
    对复数中实部和虚部的小数点部分进行数据量化
    无论几维的np.ndarray都可以输入
    """
    #print("target: ", x)
    real = np.real(x)
    imag = np.imag(x)
    real = real_convert(real, frac_bit, int_bit)
    imag = real_convert(imag, frac_bit, int_bit)
    return real + 1j * imag


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg