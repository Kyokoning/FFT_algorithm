# -*- coding: utf-8 -*-
# @Author: xnchen
# @Email: xnchen97@gmail.com
# @Time: 2021/1/10
# @File: plot.py

import numpy as np
import matplotlib.pyplot as plt
from fft import fft, DIF_FFT
from Data import DataHandler
from src import SNR, AverageMeter
from typing import List


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
