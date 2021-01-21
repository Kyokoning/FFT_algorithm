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

from plot import from_result_file_gen_snr


if __name__=='__main__':

    from_result_file_gen_snr(
        file = "补码/result_data_hard_2(1).txt",
        int_bit=3, frac_bit=6, resort_flag =True,
        cut_in=True
    )