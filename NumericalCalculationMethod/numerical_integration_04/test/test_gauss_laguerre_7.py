# -*- coding: UTF-8 -*-
"""
@file:test_gauss_laguerre_7.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from numerical_integration_04.gauss_laguerre_int import GaussLaguerreIntegration


fh = lambda x: np.sin(x) * np.exp(-x)

gli = GaussLaguerreIntegration(fh, [0, np.infty], 5)
gli.fit_int()
print("拉盖尔多项式零点：", gli.zero_points)
print("插值型求积系数：", gli.A_k)
print("积分近似值：", gli.int_value, "误差：", 0.5 - gli.int_value)

gli = GaussLaguerreIntegration(fh, [2, np.infty], 15)
gli.fit_int()
print("积分值：", gli.int_value, "误差：", 0.5 * np.exp(-2) * (np.cos(2) + np.sin(2)) - gli.int_value)

gli = GaussLaguerreIntegration(fh, [-2, np.infty], 15)
gli.fit_int()
print("积分值：", gli.int_value, "误差：", 0.5 * np.exp(2) * (np.cos(-2) + np.sin(-2)) - gli.int_value)


# 实验题目
# alpha = np.linspace(0, 1.5, 30)
# int_values = []
# for a in alpha:
#     fh = lambda x: np.sin(a ** 2 * x) * np.exp(-x)
#     gli = GaussLaguerreIntegration(fh, [0, np.infty], 10)
#     gli.fit_int()
#     int_values.append(gli.int_value)
#     print(a, gli.int_value)
#
# plt.figure(figsize=(7, 5))
# plt.plot(alpha, int_values, "ko-")
# plt.show()