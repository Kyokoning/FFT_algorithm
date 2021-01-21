# -*- coding: utf-8 -*-
# @Author: xnchen
# @Email: xnchen97@gmail.com
# @Time: 2021/1/5
# @File: src.py

import numpy as np
from typing import Tuple, List, Union
import os
import csv
import xlrd


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
                int_bit: int = 0,
                frac_bit: int = 9) -> Tuple[List[str], List[str]]:
    """
    从np.ndarray[np.complex]生成verilog测试的10位数据 补码形式（包括符号位）
    """
    real = np.real(x) * (2 ** frac_bit)
    imag = np.imag(x) * (2 ** frac_bit)

    real_int = np.array(real, dtype=np.int32)
    imag_int = np.array(imag, dtype=np.int32)

    # for x, x_int in zip(np.real(x), real_int):
    #     print(x, x_int, bin(x_int & (2**(int_bit+frac_bit+1)-1))[2:])
    res_real = [bin(x & (2 ** (int_bit + frac_bit + 1) - 1))[2:] for x in real_int]
    res_imag = [bin(x & (2 ** (int_bit + frac_bit + 1) - 1))[2:] for x in imag_int]

    res_real = ['0' * (int_bit + frac_bit + 1 - len(x)) + x for x in res_real]
    res_imag = ['0' * (int_bit + frac_bit + 1 - len(x)) + x for x in res_imag]

    return res_real, res_imag


def real_convert(x: np.ndarray,
                 frac_bit: int = 10,
                 int_bit: int = 10,
                 bin_flag: bool = False, ) -> np.ndarray:
    """
    实数的量化：小数和整数部分是分开的，
    :param x: 待转换64位数据
    :param frac_bit: 小数长度(不算符号位
    :param int_bit: 整数长度
    :param bin_flag: 用来确定是返回bin还是返回十进制 bin的格式和前人xlsx中一致，
                例子：-0.1001
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


def convert(x: np.ndarray, int_bit: int = 0,
            frac_bit: int = 9) -> Tuple[List[str], List[str]]:
    """
    将complex convert转换的内部为complex的像1.001+1j*-10.0101转换为原码
    :param x:
    :return:
    """
    real = np.real(x)
    imag = np.imag(x)

    real = real * 10 ** frac_bit
    imag = imag * 10 ** frac_bit
    real = [str(int(x)) for x in real]
    imag = [str(int(x)) for x in imag]

    for i in range(len(real)):
        if real[i][0] == "-":
            real[i] = '1' + '0' * (int_bit + frac_bit + 1 - len(real[i])) + real[i][1:]
        else:
            real[i] = '0' * (int_bit + frac_bit + 1 - len(real[i])) + real[i]

        if imag[i][0] == "-":
            imag[i] = '1' + '0' * (int_bit + frac_bit + 1 - len(imag[i])) + imag[i][1:]
        else:
            imag[i] = '0' * (int_bit + frac_bit + 1 - len(imag[i])) + imag[i]

    return real, imag


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


def read_from_txt_groundtruth(file_name: str = "output_test_data.txt",
                              N: int = 8) -> List[np.ndarray]:
    if not os.path.isfile(file_name):
        raise FileExistsError
    res = []
    cnt = 0
    with open(file_name, "r") as f:
        for line in f:
            if cnt == N:
                cnt = 0
                res.append(np.array(temp_res))
            if cnt == 0 or cnt == N:
                temp_res = []
            temp_res.append(np.complex(line.strip()))
            cnt += 1
        if cnt == N:
            res.append(np.array(temp_res))
    return res


def read_from_txt(file_name: str = "output_data.txt", N: int = 8,cut_in=False
                  ) -> Tuple[List[str], List[str]]:
    """
    从文件读入数据(二进制那种)，返回Tuple第一位是real，第二位是imag，都是二进制格式
    """
    if not os.path.isfile(file_name):
        raise FileExistsError
    cnt = 0
    real_list, imag_list = [], []
    with open(file_name, 'r') as f:
        for line in f:
            if cnt == N:
                cnt = 0
                real_list.append(temp_real)
                imag_list.append(temp_imag)
            if cnt == 0 or cnt == N:
                temp_real, temp_imag = [], []
            real, imag = line.strip().split(' ')
            if not cut_in:
                temp_real.append(real)
                temp_imag.append(imag)
            else:
                temp_real.append(real[0]+real[2:5]+real[5:11])
                temp_imag.append(imag[0]+imag[2:5]+imag[5:11])
            cnt += 1
    if cnt == N:
        real_list.append(temp_real)
        imag_list.append(temp_imag)
    return real_list, imag_list


def from_complete_list_to_float(complete_list: List[str],
                                int_bit: int = 3,
                                frac_bit: int = 6) -> np.ndarray:
    return np.array(list(map(
        lambda x: from_complete_str_to_float(x, int_bit, frac_bit),
        complete_list
    )))


def from_complete_str_to_float(complete: str, int_bit: int = 3,
                               frac_bit: int = 9) -> float:
    """
    从str格式表示的补码转换成float
    """
    if complete[0] == "0":
        return int(complete, 2) / (2 ** frac_bit)
    else:
        return -(2 ** (int_bit + frac_bit + 1) - int(complete, 2)) / (2 ** frac_bit)


def from_aux_to_float(aux_int: int, int_bit: int = 3,
                      frac_bit: int = 6) -> float:
    """
    从十进制数表示的1+3+6的原码转换成正常的小数
    :param aux_int: 例如：77,256，之类的数字
    :param int_bit:
    :param frac_bit:
    :return:
    """
    sign = bool((aux_int & (2 ** (int_bit + frac_bit))) >> (int_bit + frac_bit))
    res = aux_int - (aux_int & (2 ** (int_bit + frac_bit)))
    res = res / (2 ** frac_bit)
    return res if not sign else -res


def from_aux_list_to_float(aux_list: Union[np.ndarray, list],
                           int_bit: int = 3, frac_bit: int = 6) -> np.ndarray:
    res = list(
        map(
            lambda x: from_aux_to_float(x, int_bit, frac_bit),
            aux_list)
    )
    return np.array(res)


def from_vivado_result_read_test(file: str = "VIVADO测试结果/iladata_basic1.xlsx",
                                 start: int = 2, end: int = 1026,
                                 real_col: int = 6, imag_col: int = 7
                                 ) -> Tuple[list, list]:
    """
    从文件中读取表示虚数的实部和虚部的两个list，从start到end（excel的index-1）
    end不包在内
    """
    if not os.path.isfile(file):
        raise FileExistsError
    with xlrd.open_workbook(file) as f:
        sheet = f.sheet_by_index(0)
        """
        [text:'Sample in Buffer',
         text:'Sample in Window',
         text:'TRIGGER',
         text:'rst_n_1',
         text:'din_valid_1',
         text:'dout_valid_OBUF',
         text:'din_re[9:0]',
         text:'din_im[9:0]',
         text:'dout_re[9:0]',
         text:'dout_im[9:0]']"""
        real_unsigned = np.array([x.value for x in sheet.col_slice(real_col, start, end)],
                                 dtype=np.int32)
        imag_unsigned = np.array([x.value for x in sheet.col_slice(imag_col, start, end)],
                                 dtype=np.int32)

    return real_unsigned, imag_unsigned


def from_vivado_result_read(file: str = "VIVADO测试结果/iladata.csv"
                            ) -> Tuple[list, list, list, list]:
    """
    从文件中读取好多个64bit的输入和输出虚数。
    :param file:
    :return:
    """
    if not os.path.isfile(file):
        raise FileExistsError
    csvFile = open(file, "r")
    reader = csv.reader(csvFile)

    latency = 34
    fft_bit = 64
    sleep_clock = 30  # 中间无论输入输出都无效
    input_real_unsigned = []
    input_imag_unsigned = []
    output_real_unsigned = []
    output_imag_unsigned = []
    start_line = 2
    cnt = 0

    input_real_unsigned_temp = []
    input_imag_unsigned_temp = []
    output_real_unsigned_temp = []
    output_imag_unsigned_temp = []
    for item in reader:
        if reader.line_num < start_line + 1:
            continue
        if cnt < latency:
            # 在第一个阶段，只有数字输入
            input_real_unsigned_temp.append(int(item[6]))
            input_imag_unsigned_temp.append(int(item[7]))
            cnt += 1

        elif cnt < fft_bit:
            # 第二个阶段，数字和fft一起输入
            input_real_unsigned_temp.append(int(item[6]))
            input_imag_unsigned_temp.append(int(item[7]))
            output_real_unsigned_temp.append(int(item[8]))
            output_imag_unsigned_temp.append(int(item[9]))
            cnt += 1

        elif cnt < fft_bit + latency:
            # 第三个阶段，只有fft输入
            output_real_unsigned_temp.append(int(item[8]))
            output_imag_unsigned_temp.append(int(item[9]))
            cnt += 1

        elif cnt < fft_bit + latency + sleep_clock - 1:
            # 第四个阶段：睡眠
            cnt += 1

        else:
            cnt = 0
            input_real_unsigned.append(input_real_unsigned_temp)
            input_imag_unsigned.append(input_imag_unsigned_temp)
            output_real_unsigned.append(output_real_unsigned_temp)
            output_imag_unsigned.append(output_imag_unsigned_temp)
            input_real_unsigned_temp = []
            input_imag_unsigned_temp = []
            output_real_unsigned_temp = []
            output_imag_unsigned_temp = []
    return input_real_unsigned, input_imag_unsigned, \
           output_real_unsigned, output_imag_unsigned


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
