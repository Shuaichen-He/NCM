# -*- coding: UTF-8 -*-
"""
@file_name: test_newton_cotes_int_1.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from numerical_integration_04.newton_cotes_integration import NewtonCotesIntegration
from util_font import *

# 例1
int_fun = lambda x: np.exp(-0.5 * x) * np.sin(x + np.pi / 6)

num = np.linspace(2, 9, 8)  # 划分区间数
int_res = np.zeros(len(num))  # 存储各划分区间数下的积分值
for i, n in enumerate(num):
    nci = NewtonCotesIntegration(int_fun, int_interval=[0, 3 * np.pi], interval_num=n)  # 实例化对象
    nci.fit_cotes_int()  # 求解积分
    print("n = %d，科特斯系数：\n" % n, nci.cotes_coefficient)  # 打印科特斯系数
    int_res[i] = nci.int_value  # 存储积分值
print("各划分区间数下的积分误差：\n", 0.900840787818886 - int_res)  # 误差计算

# 可视化被积函数积分区域
plt.figure(figsize=(7, 5))
xi = np.linspace(0, 3 * np.pi, 100)
yi = int_fun(xi)
plt.plot(xi, yi, "k--", lw=0.5)  # 描线
plt.fill_between(xi, yi, color="c", alpha=0.5)  # 填充区域
plt.xlabel(r"$x$", fontdict={"fontsize": 18})
plt.ylabel(r"$f(x)$", fontdict={"fontsize": 18})
plt.title("被积函数$f(x)$的积分区域", fontdict={"fontsize": 18})
plt.text(2.5, 0.5, r"$f(x)=e^{-0.5x}sin \left(x+\dfrac{\pi}{6} \right) \quad x \in [0,3\pi]$", fontsize=18)
plt.tick_params(labelsize=16)  # 刻度字体大小16
plt.show()
