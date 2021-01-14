# -*- coding: utf-8 -*-
# @Author: xnchen
# @Email: xnchen97@gmail.com
# @Time: 2021/1/5
# @File: src.py

import numpy as np
from typing import Tuple, List
import os


def SNR(signal: np.ndarray, gt: np.ndarray) -> float:
    signal = np.asanyarray(signal)
    gt = np.asanyarray(gt)
    error = np.sum(np.abs(signal - gt) ** 2)
    output = np.sum(np.abs(gt) ** 2)
    snr = 10 * np.log10(output / error)
    return snr


def convert_to_fix(arr: np.ndarray, bit: int, bin_flag: bool = False) -> np.ndarray:
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
    if bin_flag:
        return from_dec_list_to_binary(res.astype(np.int64))
    else:
        return res.astype(np.int64)


def convert_to_float(arr: np.ndarray, bit: int, bin_flag: bool = False) -> np.ndarray:
    """
    将 convert_to_fix的Array[int]格式的量化数字再转成Array[np.float]
    """
    if bin_flag:
        return arr / (10 ** bit)
    else:
        return arr / (2 ** bit)


def fix_fractional_part(x: np.ndarray, bit: int, bin_flag: bool = False) -> np.ndarray:
    """
    小数的量化
    """
    return convert_to_float(convert_to_fix(x, bit, bin_flag), bit, bin_flag)


def fix_integer_part(x: np.ndarray, bit: int, bin_flag: bool = False) -> np.ndarray:
    """
    整数的量化
    """
    # print("\tx: ",x,
    #      "\n\tinteger_fix: ",np.clip(x, -2 ** bit + 1, 2 ** bit - 1))
    res = np.array(np.clip(x, -2 ** bit + 1, 2 ** bit), dtype=np.int32)
    if bin_flag:
        return from_dec_list_to_binary(res)
    else:
        return res


def from_dec_list_to_binary(x: np.ndarray) -> np.ndarray:
    """
    把ndarray包裹的十进制int转换成二进制并且是数字int形式
    """
    return np.array([int(np.binary_repr(val)) for val in x])

def verilog_res(x: np.ndarray,
                int_bit :int=0,
                frac_bit:int=9)->Tuple[List[str], List[str]]:
    """生成verilog测试的10位数据（包括符号位）"""
    assert int_bit+frac_bit==9
    real = np.real(x)*(2**frac_bit)
    imag = np.imag(x)*(2**frac_bit)

    real_int = np.array(real, dtype=np.int32)
    imag_int = np.array(imag, dtype = np.int32)


    for x, x_int in zip(np.real(x), real_int):
        print(x, x_int, bin(x_int & (2**(int_bit+frac_bit+1)-1))[2:])
    res_real = [bin(x & (2**(int_bit+frac_bit+1)-1))[2:] for x in real_int]
    res_imag = [bin(x & (2**(int_bit+frac_bit+1)-1))[2:] for x in imag_int]

    res_real = ['0'*(int_bit+frac_bit+1-len(x))+x for x in res_real]
    res_imag = ['0'*(int_bit+frac_bit+1-len(x))+x for x in res_imag]

    return res_real, res_imag


def real_convert(x: np.ndarray,
                 frac_bit: int = 10,
                 int_bit: int = 10,
                 bin_flag: bool = False,
                 complement: bool = False) -> np.ndarray:
    """
    实数的量化：小数和整数部分是分开的，
    :param x: 待转换64位数据
    :param frac_bit: 小数长度(不算符号位
    :param int_bit: 整数长度
    :param bin_flag: 用来确定是返回bin还是返回十进制 bin的格式和前人xlsx中一致，
                例子：-0.1001
    :param complement: 置True返回补码格式QAQ 在bin_flag=True的时候生效
    """
    int_part, frac_part = divide_int_frac(x)
    int_part = fix_integer_part(int_part, int_bit, bin_flag)
    frac_part = fix_fractional_part(frac_part, frac_bit, bin_flag)
    return merge_int_frac(int_part, frac_part)


def merge_int_frac(int_part: np.ndarray, frac_part: np.ndarray) -> np.ndarray:
    return int_part + frac_part


def divide_int_frac(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    将array中的数字的整数和小数部分分开
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
                    int_bit: int = 10,
                    frac_bit: int = 10,
                    bin_flag: bool = False) -> np.ndarray:
    """
    对复数中实部和虚部的小数点部分进行数据量化
    无论几维的np.ndarray都可以输入
    """
    # print("target: ", x)
    real = np.real(x)
    imag = np.imag(x)
    real = real_convert(real, frac_bit, int_bit, bin_flag)
    imag = real_convert(imag, frac_bit, int_bit, bin_flag)
    return real + 1j * imag



def complex_convert_list(x: List[np.ndarray],
                         int_bit: int = 10,
                         frac_bit: int = 10,
                         bin_flag: bool = False) -> List[np.ndarray]:
    """
    将List包裹的内容为64位采样的复数进行数据量化
    """
    return list(
        map(lambda i: complex_convert(i, int_bit, frac_bit, bin_flag), x)
    )


def from_divide_list_to_complex(real: List[np.ndarray], imag: List[np.ndarray]
                                ) -> List[np.ndarray]:
    return [a + 1j * b for a, b in zip(real, imag)]

def read_from_txt(file_name:str = "output_data.txt", N:int=8
                  )->Tuple[List[str], List[str]]:
    """
    从文件读入数据(二进制那种)，返回Tuple第一位是real，第二位是imag，都是二进制格式
    """
    if not os.path.isfile(file_name):
        raise FileExistsError
    real_list, imag_list = [], []
    cnt = 0
    with open(file_name, 'r') as f:
        for line in f:
            if cnt==N:
                cnt = 0
                real_list.append(temp_real)
                imag_list.append(temp_imag)
            if cnt == 0 or cnt==N:
                temp_real, temp_imag = [], []
            real, imag = line.strip().split(' ')
            temp_real.append(real)
            temp_imag.append(imag)
            cnt += 1
    if cnt == N:
        real_list.append(temp_real)
        imag_list.append(temp_imag)
    return real_list, imag_list

def from_complete_list_to_float(complete_list: List[str],
                                int_bit: int=3,
                                frac_bit: int=3)->np.ndarray:
    return np.array(list(map(
        lambda x:from_complete_str_to_float(x, int_bit, frac_bit),
        complete_list
    )))

def from_complete_str_to_float(complete: str, int_bit: int=3,
                               frac_bit:int=9)->float:
    """
    从str格式表示的补码转换成float
    """
    if complete[0]=="0":
        return int(complete, 2)/(2**frac_bit)
    else:
        return -(2**(int_bit+frac_bit+1)-int(complete, 2))/(2**frac_bit)


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
