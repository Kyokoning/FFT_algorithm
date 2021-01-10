# -*- coding: utf-8 -*-
# @Author: xnchen
# @Email: xnchen97@gmail.com
# @Time: 2021/1/6
# @File: main.py

import numpy as np
import matplotlib.pyplot as plt
from fft import fft, DIF_FFT
from Data import DataHandler
from src import SNR, AverageMeter, \
    from_divide_list_to_complex, complex_convert

d = DataHandler()
d.input_from_excel()
input = from_divide_list_to_complex(d.test_input_real, d.test_input_imag)
output_float = from_divide_list_to_complex(d.test_output_real, d.test_output_imag)

res = complex_convert(input[0], bin_flag=True)
