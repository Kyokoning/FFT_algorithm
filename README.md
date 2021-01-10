### Abstract

这是用来生成FFT硬件电路测试数据的python脚本

### 文件

|名字|用途| |--| -- | |Data.py|数据读取类，随机生成数据/从前人留下的xlsx中读取数据都用这个里面的类DataHandler做| |fft.py|内含DIF和DIT的基2FFT实现方法|
|main.py|运行入口，目前用来测试| |plot.py|数据量化中，进行不同位数据SNR的测试方法| |src.py|包括：量化、SNR等方法|

### 常用操作示例

**随机数据生成**：

```python
from Data import DataHandler

d = DataHandler()
d.gen_data(20, real=True)  # 随机生成20个数据，并且其中只有real部分（虚数部分置零）
# 生成的数据是List[np.array[np.float32]]格式
# d.test_input_real
# d.test_input_imag
# d.test_output_real
# d.test_output_imag
```

**从xlsx格式文件中读取**

```python
from Data import DataHandler

d = DataHandler()
d.input_from_excel()  # 该excel只有一个64个采样的数据，因此我们的结果只增加一个
# 生成的数据是List[np.array[np.float32]]格式
# d.test_input_real
# d.test_input_imag
# d.test_output_real
# d.test_output_imag
```

**进行量化**

```python
from Data import DataHandler
from src import from_divide_list_to_complex, complex_convert_list

d = DataHandler()
d.gen_data(20)

# 将两个内部分别存储实数和虚数部分的数据合并，最内层数据为np.complex64
input = from_divide_list_to_complex(d.test_input_real, d.test_input_imag)
output = from_divide_list_to_complex(d.test_output_real, d.test_output_imag)

# complex_convert_list方法用于量化，int_bit是整数部分的位数，frac_bit是小数部分的位数，
# bin_flag代表了是否要得到二进制结果（二进制结果的显示参考.DFT64test.xlse，例子：-1101.010)
input_quant = complex_convert_list(x=input,
                                   int_bit=7,
                                   frac_bit=7,
                                   bin_flag=True)
output_quant = complex_convert_list(x=output,
                                    int_bit=7,
                                    frac_bit=7,
                                    bin_flag=True)
```

**FFT**

```python
from Data import DataHandler
from src import from_divide_list_to_complex, complex_convert_list, SNR
from fft import DIF_FFT

d = DataHandler()
d.gen_data(20)

# 将两个内部分别存储实数和虚数部分的数据合并，最内层数据为np.complex64
input = from_divide_list_to_complex(d.test_input_real, d.test_input_imag)
output = from_divide_list_to_complex(d.test_output_real, d.test_output_imag)

# DIF_FFT是8位基2FFT的方法，frac_bit和int_bit分别用于控制内部的量化位数
# frac_bit只可以用固定位数/None，int_bit可以输入数字/七个数字的List/None
# SNR是算信噪比的方法
for data, groundtruth in zip(input, output):
    dif_res = DIF_FFT(
        data, frac_bit=7, int_bit=7
    )
    # 或者
    dif_res = DIF_FFT(
        data, frac_bit=7, int_bit=[4, 5, 6, 7, 7, 7, 7]
    )
    print("SNR: ", SNR(signal=dif_res, gt=groundtruth))
```




