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

def from_basic_result_file_gen_snr():
    """
    从basic的vivado输出文件（竟然还是excel）得到snr
    每个文件只有一个数据请注意
    :return:
    """
    start=326
    end=326+64+1

    input_real_unsigned, input_imag_unsigned = from_vivado_result_read_test(
        file = "VIVADO测试结果/iladata_basic2.xlsx",
        start=start, end=end
    )
    output_real_unsigned, output_imag_unsigned = from_vivado_result_read_test(
        file = "VIVADO测试结果/iladata_basic2.xlsx",
        start=514, end=578, real_col=8, imag_col=9
    )
    fft_bit = 64
    f1 = open("VIVADO测试结果/随机输入_basic.txt", 'w')
    f2 = open("VIVADO测试结果/输出结果_basic.txt", 'w')
    f3 = open("VIVADO测试结果/使用python跑的输出_basic.txt", 'w')
    times = end-64-start

    output_real = from_aux_list_to_float(output_real_unsigned,
                                         int_bit=5,
                                         frac_bit=4)
    output_imag = from_aux_list_to_float(output_imag_unsigned,
                                         int_bit=5,
                                         frac_bit=4)

    output = output_real + 1j * output_imag
    for j in range(64):
        f2.writelines(str(output[j])+'\n')
    print("=====================input==================")
    for i in range(times):
        input_real = from_aux_list_to_float(
            input_real_unsigned[i:(i+64)],
            int_bit=0, frac_bit=9
        )
        input_imag = from_aux_list_to_float(
            input_imag_unsigned[i:(i+64)],
            int_bit=0, frac_bit=9)
        input = input_real + 1j * input_imag

        gt = np.fft.fft(input)
        # for j in range(64):
        #     print(gt[j])



        for j in range(64):
            f1.writelines(str(input[j]) + "\n")
            # f2.writelines(str(output[j]) + '\n')
            f3.writelines(str(gt[j]) + "\n")

        snr_single = SNR(output, gt)
        print("start with index ", i+start, '\tSNR = ', snr_single)
    # f1.close()
    f2.close()
    # f3.close()


def from_result_file_gen_snr(file:str="补码/result_data_hard.txt",
                             int_bit:int=3,
                             frac_bit:int=6,
                             resort_flag:bool = False,
                             cut_in:bool=False):
    """
    从他们更新后的verilog仿真文件得到硬件的snr
    :return:
    """
    from fft import resort
    pred_real, pred_imag = read_from_txt(file, cut_in=cut_in)
    gt = read_from_txt_groundtruth("output_test_data.txt")
    snr = AverageMeter()
    print(pred_imag)
    for i in range(len(pred_real)):

        pred_real_float = from_complete_list_to_float(pred_real[i], int_bit, frac_bit)
        pred_imag_float = from_complete_list_to_float(pred_imag[i], int_bit, frac_bit)
        if resort_flag:
            res = resort(pred_real_float+1j*pred_imag_float)
        else:
            res = pred_real_float+1j*pred_imag_float
        temp_snr = SNR(signal = res,
                       gt = gt[i])
        snr.update(temp_snr)
        print(i, '\tsingle SNR: ', temp_snr, "\tAverage SNR: ",snr.average())

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
