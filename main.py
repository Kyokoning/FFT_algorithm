# -*- coding: utf-8 -*-
# @Author: xnchen
# @Email: xnchen97@gmail.com
# @Time: 2021/1/6
# @File: main.py

import numpy as np
import matplotlib.pyplot as plt
from fft import fft, DIF_FFT
from Data import DataHandler
from src import read_from_txt, \
    from_complete_list_to_float, AverageMeter, SNR,\
    read_from_txt_groundtruth

pred_real, pred_imag = read_from_txt("result_data_hard.txt")
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


