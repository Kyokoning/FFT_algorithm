# -*- coding: utf-8 -*-
# @Author: xnchen
# @Email: xnchen97@gmail.com
# @Time: 2021/1/6
# @File: main.py

import numpy as np
import matplotlib.pyplot as plt
from fft import fft, DIF_FFT
from Data import DataHandler
import os
from src import *
from fft import DIF_stage1, DIF_stage2, DIF_stage3, resort

from plot import from_vivado_result_read
start=325
end=578

input_real_unsigned, input_imag_unsigned = from_vivado_result_read_test(
    file = "VIVADO测试结果/iladata_basic2.xlsx",
    start=start, end=end
)
output_real_unsigned, output_imag_unsigned = from_vivado_result_read_test(
    file = "VIVADO测试结果/iladata_basic2.xlsx",
    start=514, end=578, real_col=8, imag_col=9
)
fft_bit = 64
# f1 = open("VIVADO测试结果/随机输入_basic.txt", 'w')
f2 = open("VIVADO测试结果/输出结果_basic.txt", 'w')
# f3 = open("VIVADO测试结果/使用python跑的输出_basic.txt", 'w')
times = end-64-start

output_real = from_aux_list_to_float(output_real_unsigned,
                                     int_bit=6,
                                     frac_bit=3)
output_imag = from_aux_list_to_float(output_imag_unsigned,
                                     int_bit=6,
                                     frac_bit=3)

output = output_real + 1j * output_imag
# for j in range(64):
#     f2.writelines(str(output[j])+'\n')
print("=====================input==================")
for i in range(times):
    input_real = from_aux_list_to_float(input_real_unsigned[i:(i+64)], int_bit=0, frac_bit=9)
    input_imag = from_aux_list_to_float(input_imag_unsigned[i:(i+64)], int_bit=0, frac_bit=9)
    input = input_real + 1j * input_imag

    gt = np.fft.fft(input)
    # for j in range(64):
    #     print(gt[j])



    # for j in range(fft_bit):
    #     f1.writelines(str(input[j]) + "\n")
    #     f2.writelines(str(output[j]) + '\n')
    #     f3.writelines(str(gt[j]) + "\n")

    snr_single = SNR(output, gt*2)
    print("start with index ", i+start, '\tSNR = ', snr_single)
# f1.close()
f2.close()
# f3.close()