# -*- coding: UTF-8 -*-
"""
@file_name: test_FFT.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import matplotlib.pyplot as plt
from function_approximation_03.fast_fourier_transform import FastFourierTransformApproximation


# 例7示例
# fun = lambda x: x ** 4 - 3 * x ** 3 + 2 * x ** 2 - np.tan(x * (x - 2))
# x = np.linspace(0, 2, 8, endpoint=False)
#
# p_num = [32, 64]  # 离散点数
# plt.figure(figsize=(14, 5))
# for i, p in enumerate(p_num):
#     x = np.linspace(0, 2, p, endpoint=False)
#     fft = FastFourierTransformApproximation(y=fun(x), interval=[0, 2], fun=fun)
#     fft.fit_fourier()
#     print("正弦项系数：\n", fft.Bk)
#     print("余弦项系数：\n", fft.Ak)
#     print("傅里叶逼近多项式：\n", fft.approximation_poly)
#     plt.subplot(121 + i)
#     fft.plt_approximate(is_show=False, is_fh_marker=True)  # 可视化
# plt.show()
#
# # 例9代码
fun2 = lambda x: 0.5 * x ** 2 * np.cos(4 * x) + np.sin(x ** 2)
# fun2 = lambda x: np.cos(np.pi * x) - 2 * np.sin(np.pi * x)
plt.figure(figsize=(14, 5))
x = np.linspace(-np.pi, np.pi, 16, endpoint=False)
fft = FastFourierTransformApproximation(y=fun2(x), interval=[-np.pi, np.pi], fun=fun2)
fft.fit_fourier()
plt.subplot(121)
fft.plt_approximate(is_show=False, is_fh_marker=False)  # 可视化
x = np.linspace(-np.pi, np.pi, 64, endpoint=False)
fft = FastFourierTransformApproximation(y=fun2(x), interval=[-np.pi, np.pi], fun=fun2)
fft.fit_fourier()
plt.subplot(122)
fft.plt_approximate(is_show=False, is_fh_marker=True)  # 可视化
plt.show()