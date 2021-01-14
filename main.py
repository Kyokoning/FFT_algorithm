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
    from_complete_list_to_float, AverageMeter, SNR

pred_real, pred_imag = read_from_txt("result_data_hard.txt")
gt_real, gt_imag = read_from_txt("result_data.txt")

snr = AverageMeter()
for i in range(len(gt_real)):
    pred_real_float = from_complete_list_to_float(pred_real[i])
    pred_imag_float = from_complete_list_to_float(pred_imag[i])
    gt_real_float = from_complete_list_to_float(pred_real[i])
    gt_imag_float = from_complete_list_to_float(gt_imag[i])
    temp_snr = SNR(signal = pred_real_float+1j*pred_imag_float,
            gt = gt_real_float+1j*gt_imag_float)
    snr.update(temp_snr)
    print(i, '\t', temp_snr, snr.average())


