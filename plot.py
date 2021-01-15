# -*- coding: utf-8 -*-
# @Author: xnchen
# @Email: xnchen97@gmail.com
# @Time: 2021/1/10
# @File: plot.py

import numpy as np
import matplotlib.pyplot as plt
from fft import fft, DIF_FFT
from Data import DataHandler
from src import *
from typing import List
import os
from fft import *




def from_file_gen_fft_3_stage_test_file(dir:str = "fft_8bit_stage_divide"):
    """
    从gen_verilog_test_file()方法生成的文件中得到基-2 8bit FFT文件的方法
    :return:
    """
    input = read_from_txt_groundtruth(file_name="input_test_data.txt")
    fft_bit = 8
    if not os.path.isdir(dir):
        os.mkdir(dir)
    fft1_f, fft2_f, fft3_f, result_f = os.path.join(dir, "fft_1"), \
                                       os.path.join(dir, "fft_2"), \
                                       os.path.join(dir, "fft_3"), \
                                       os.path.join(dir, "result")
    fft1_number_f, fft2_number_f, fft3_number_f = os.path.join(dir, "fft_test_1"), \
                                                  os.path.join(dir, "fft_test_2"), \
                                                  os.path.join(dir, "fft_test_3")
    with open(fft1_f, 'w') as f1, open(fft2_f, "w") as f2, \
            open(fft3_f, "w") as f3, open(result_f, "w") as fres, \
            open(fft1_number_f, 'w') as f1_number,open(fft2_number_f, 'w') as f2_number, \
            open(fft3_number_f, 'w') as f3_number:
        for i in range(len(input)):
            fft1 = np.array(input[i])
            DIF_stage1(fft1)
            fft1_real, fft1_imag = verilog_res(fft1, int_bit=2, frac_bit=9)

            fft2 = np.array(fft1)
            DIF_stage2(fft2)
            fft2_real, fft2_imag = verilog_res(fft2, int_bit=3, frac_bit=9)

            fft3 = np.array(fft2)
            DIF_stage3(fft3)
            fft3_real, fft3_imag = verilog_res(fft3, int_bit=4, frac_bit=9)

            res = resort(fft3)
            res_real, res_imag = verilog_res(res, int_bit = 4, frac_bit=9)
            for j in range(fft_bit):
                f1.writelines(fft1_real[j]+" "+fft1_imag[j]+"\n")
                f2.writelines(fft2_real[j]+" "+fft2_imag[j]+"\n")
                f3.writelines(fft3_real[j]+" "+fft3_imag[j]+"\n")
                fres.writelines(res_real[j]+" "+res_imag[j]+"\n")
                f1_number.writelines(str(fft1[j])+"\n")
                f2_number.writelines(str(fft2[j])+"\n")
                f3_number.writelines(str(fft3[j])+"\n")


def from_result_file_gen_snr():
    """
    从verilog仿真文件得到硬件的snr
    :return:
    """
    pred_real, pred_imag = read_from_txt("补码/result_data_hard.txt")
    gt = read_from_txt_groundtruth("output_test_data.txt")
    snr = AverageMeter()
    for i in range(len(gt)):
        pred_real_float = from_complete_list_to_float(pred_real[i])
        pred_imag_float = from_complete_list_to_float(pred_imag[i])
        #print(pred_real_float+1j*pred_imag_float)
        temp_snr = SNR(signal = pred_real_float+1j*pred_imag_float,
                       gt = gt[i])
        snr.update(temp_snr)
        print(i, '\t', temp_snr, snr.average())

def gen_verilog_test_file(data_num:int=1, fft_bit:int=8,
                          dir:str = "补码/"):
    """
    生成补码表示的测试verilog文件
    :param data_num:
    :param fft_bit:
    :return:
    """
    if not os.path.isdir(dir):
        os.mkdir(dir)
    d = DataHandler()
    d.gen_data(data_num, fft_bit)

    input_list, output_list = d.from_divide_list_to_complex()
    input_f = open(os.path.join(dir, "补码/input_data.txt"), 'w')
    output_f = open(os.path.join(dir, "补码/result_data.txt"), 'w')
    input_test_f = open(os.path.join(dir, "input_test_data.txt"), 'w')
    output_test_f = open(os.path.join(dir, "output_test_data.txt"), 'w')
    for i in range(data_num):

        input_real_bit, input_imag_bit = verilog_res(input_list[i], 0, 9)
        output_real_bit, output_imag_bit = verilog_res(output_list[i], 3, 6)
        for j in range(fft_bit):
            input_test_f.writelines(str(input_list[i][j]) + '\n')
            output_test_f.writelines(str(output_list[i][j]) + '\n')
            input_f.writelines(input_real_bit[j] + " " + input_imag_bit[j] + "\n")
            output_f.writelines(output_real_bit[j] + " " + output_imag_bit[j] + "\n")
    input_f.close()
    output_f.close()
    output_test_f.close()
    input_test_f.close()

def gen_verilog_test_file_aux(data_num:int = 1, fft_bit:int=8,
                              dir:str="原码/"):
    """
    生成原码表示的测试verilog文件
    :param data_num:
    :param fft_bit:
    :return:
    """
    if not os.path.isdir(dir):
        os.mkdir(dir)
    d = DataHandler()
    d.gen_data(data_num, fft_bit)

    input_list, output_list = d.from_divide_list_to_complex()
    input_f = open(os.path.join(dir, "补码/input_data.txt"), 'w')
    output_f = open(os.path.join(dir, "补码/result_data.txt"), 'w')
    input_test_f = open(os.path.join(dir, "input_test_data.txt"), 'w')
    output_test_f = open(os.path.join(dir, "output_test_data.txt"), 'w')
    for i in range(data_num):
        input = complex_convert(input_list[i], 0, 9)
        output = complex_convert(output_list[i], 3, 6)
        input_real_bit, input_imag_bit = convert(input, 0, 9)
        output_real_bit, output_imag_bit = convert(output, 3, 6)
        for j in range(fft_bit):
            input_test_f.writelines(str(input_list[i][j]) + '\n')
            output_test_f.writelines(str(output_list[i][j]) + '\n')
            input_f.writelines(input_real_bit[j] + " " + input_imag_bit[j] + "\n")
            output_f.writelines(output_real_bit[j] + " " + output_imag_bit[j] + "\n")
    input_f.close()
    output_f.close()
    output_test_f.close()
    input_test_f.close()


def test_fraction_bit(input: List[np.ndarray], fft_res: List[np.ndarray]) -> None:
    """
    在整数宽度相同的情况下，测试不同的小数宽度造成的SNR差别（并且画图）
    """
    res = []
    for i in range(2, 18):
        snr = AverageMeter()
        for j, data in enumerate(input):
            dif_res = DIF_FFT(
                data,
                frac_bit=i,
                int_bit=7
            )
            snr.update(SNR(signal=dif_res, gt=fft_res[j]))
        res.append(snr.average())
        print("bit: ", i, "\tSNR: ", snr.average())

    plt.plot(np.arange(2, 18), res, linestyle=':', label='SNR')
    plt.grid(axis='x', color='0.95', )
    plt.grid(axis='y', color='0.95')
    plt.xlabel("Fraction bit")
    plt.ylabel("SNR(dB)")
    plt.show()


def test_integer_bit(input: List[np.ndarray], fft_res: List[np.ndarray]) -> None:
    """
    在小数位宽相同的情况下，测试不同的整数位宽组合对于SNR的影响。
    """
    int_bits = [7,
                [5, 6, 7, 7, 7, 7, 7],
                [4, 5, 6, 7, 7, 7, 7],
                [4, 5, 6, 6, 7, 7, 7],
                [4, 5, 6, 6, 6, 7, 7],
                [4, 5, 6, 6, 6, 6, 7],
                [4, 5, 5, 6, 6, 7, 7],
                [4, 5, 5, 5, 6, 7, 7],
                [4, 5, 5, 5, 6, 6, 7],
                [4, 4, 5, 6, 7, 7, 7],
                [4, 4, 4, 5, 6, 7, 7],
                [3, 4, 5, 6, 7, 7, 7],
                [2, 3, 4, 5, 6, 7, 7],
                [2, 3, 4, 5, 6, 6, 7],
                [2, 3, 4, 5, 5, 6, 7],
                [2, 3, 3, 4, 5, 6, 7],
                [2, 2, 3, 4, 5, 6, 7],
                [1, 2, 3, 4, 5, 6, 7],
                ]
    # int_bits = [[2, 3, 4, 5, 6, 6, 7]]
    for int_bit in int_bits:
        snr = AverageMeter()
        for j, data in enumerate(input):
            dif_res = DIF_FFT(
                data, frac_bit=9,
                int_bit=int_bit
            )
            snr.update(SNR(signal=dif_res, gt=fft_res[j]))
        print("int_bit: ", int_bit, "\tSNR: ", snr.average())
