# -*- coding: UTF-8 -*-
"""
@file_name: exp2.py
@time: 2023-02-10
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from numerical_integration_04.gauss_laguerre_int import GaussLaguerreIntegration
from numerical_integration_04.gauss_legendre_int import GaussLegendreIntegration
from Experiment.util_font import *  # 导入字体文件

# (1) 高斯—拉盖尔求积公式
alpha = np.linspace(0, 1.5, 20)
fun = lambda x, alpha:  np.exp(-x) * np.sin(alpha ** 2 * x)
int_values = []

for alpha_ in alpha:
    int_fun = lambda x: fun(x, alpha_)
    gli = GaussLaguerreIntegration(int_fun, [0, np.infty], 10)
    gli.fit_int()
    int_values.append(gli.int_value)

# 可视化
plt.figure(figsize=(14, 5))
plt.subplot(121)
plt.plot(alpha, int_values, "-o")
plt.xlabel(r"$\alpha$", fontdict={"fontsize": 18})
plt.ylabel(r"$I(\alpha)$", fontdict={"fontsize": 18})
plt.title(r"高斯—拉盖尔：$I(\alpha)$与$\alpha$关系曲线", fontdict={"fontsize": 18})
plt.tick_params(labelsize=16)  # 刻度字体大小16
plt.grid(ls=":")

# (2) 高斯—勒让德求积公式
alpha = np.linspace(0, 5, 50)
fun = lambda x, alpha:  x ** 2 * np.sin(alpha ** 2 * x)
int_values = []
for alpha_ in alpha:
    int_fun = lambda x: fun(x, alpha_)
    gli = GaussLegendreIntegration(int_fun, [1, 2], 10)
    gli.fit_int()
    int_values.append(gli.int_value)

# 可视化
plt.subplot(122)
plt.plot(alpha, int_values, "-o")
plt.xlabel(r"$\alpha$", fontdict={"fontsize": 18})
plt.ylabel(r"$I(\alpha)$", fontdict={"fontsize": 18})
plt.title(r"高斯—勒让德：$I(\alpha)$与$\alpha$关系曲线", fontdict={"fontsize": 18})
plt.tick_params(labelsize=16)  # 刻度字体大小16
plt.grid(ls=":")
plt.show()