# -*- coding: UTF-8 -*-
"""
@file_name: exp2.py
@time: 2023-02-10
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import matplotlib.pyplot as plt
from function_approximation_03.fast_fourier_transform import FastFourierTransformApproximation

fh = lambda x: 0.5 * x ** 2 * np.cos(4 * x) + np.sin(x ** 2)
plt.figure(figsize=(14, 5))
x = np.linspace(-np.pi, np.pi, 16, endpoint=False)
fft = FastFourierTransformApproximation(y=fh(x), interval=[-np.pi, np.pi], fun=fh)
fft.fit_fourier()
print(fft.Ak)  # 展开后的余弦项系数
print(fft.Bk)  # 展开后的正弦项系数
plt.subplot(121)
fft.plt_approximate(is_show=False, is_fh_marker=False)  # 可视化
x = np.linspace(-np.pi, np.pi, 64, endpoint=False)
fft = FastFourierTransformApproximation(y=fh(x), interval=[-np.pi, np.pi], fun=fh)
fft.fit_fourier()
plt.subplot(122)
fft.plt_approximate(is_show=False, is_fh_marker=True)  # 可视化
plt.show()
