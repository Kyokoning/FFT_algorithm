# -*- coding: utf-8 -*-
# @Author: xnchen
# @Email: xnchen97@gmail.com
# @Time: 2021/1/5
# @File: Data.py

import xlrd
import numpy as np
from typing import List, Union
import matplotlib.pyplot as plt
from fft import fft
import csv


class DataHandler(object):
    def __init__(self):
        super().__init__()
        self.test_input_real = []
        self.test_input_imag = []
        self.test_output_real = []
        self.test_output_imag = []

    def gen_data(self, data_num: int = 1, N: int = 64, real: bool = False):
        for _ in range(data_num):
            self.random_data(real=real, N = N)

    def random_data(self, real=False, N: int = 64):
        """
        随机地生成一个64位数据加入list中
        """
        input_real = np.random.rand(N) * 2 - 1
        if not real:
            input_imag = np.random.rand(N) * 2 - 1
        else:
            input_imag = np.zeros(N)
        input = [np.complex(a, b) for a, b in zip(input_real, input_imag)]
        output_real, output_imag = self.fft_result(input)
        self.test_input_real.append(np.array(input_real))
        self.test_input_imag.append(np.array(input_imag))
        self.test_output_real.append(np.array(output_real))
        self.test_output_imag.append(np.array(output_imag))

    def from_divide_list_to_complex(self):
        return [a + 1j * b for a, b in zip(self.test_input_real, self.test_input_imag)], \
               [a + 1j * b for a, b in zip(self.test_output_real, self.test_output_imag)]

    def input_from_excel(self):
        """
        从前人的excel中提取一个64位数据
        """
        with xlrd.open_workbook("DFT64test.xlsx") as f:
            length = len(self.read_col_from_excel(f, 1))
            self.test_input_real.append(self.read_col_from_excel(f, 1))
            self.test_input_imag.append(np.zeros(length))
            self.test_output_real.append(self.read_col_from_excel(f, 2))
            self.test_output_imag.append(self.read_col_from_excel(f, 3))

    def read_col_from_excel(self, workbook, col: int = 1) -> list:
        sheet = workbook.sheet_by_index(0)

        col_value = sheet.col_slice(col, 1)
        col_value = np.array([x.value for x in col_value])
        return col_value

    def fft_result(self, input: List[complex]) -> List[float]:
        res = np.fft.fft(input)
        return [np.real(res), np.imag(res)]

    def plot(self):
        """
        直方图累计
        """
        for _ in range(1000000):
            self.random_data()

        x_list = [x - 20 for x in np.arange(0, 40, 0.2)]
        hist_real, k = np.histogram(self.test_output_real, x_list)
        hist_imag, _ = np.histogram(self.test_output_imag, x_list)
        x_list = [x - 0.1 for x in x_list[1:]]
        print(len(self.test_output_real), len(self.test_output_imag))
        print(sum(hist_real), sum(hist_imag))
        # 实数
        plt.subplot(211)
        plt.plot(x_list, hist_real, linestyle=':', label='real(output)')
        plt.legend(title='Parameter where:')
        plt.grid(axis='x', color='0.95', )
        plt.grid(axis='y', color='0.95')

        # 虚数
        plt.subplot(212)
        plt.plot(x_list, hist_imag, linestyle=':', label='imag(output)')
        plt.legend(title='Parameter where:')
        plt.grid(axis='x', color='0.95', )
        plt.grid(axis='y', color='0.95')


if __name__ == "__main__":
    d = DataHandler()
    d.input_from_excel()
    input = [np.complex(i, j) for i, j in zip(d.test_input_real[0], d.test_input_imag[0])]
    res = fft(input)

    print(np.allclose(res, np.fft.fft(input)))
