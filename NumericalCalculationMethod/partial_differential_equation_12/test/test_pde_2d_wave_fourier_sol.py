# -*- coding: UTF-8 -*-
"""
@file_name: test_pde_2d_wave_fourier_sol.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import matplotlib.pyplot as plt
from partial_differential_equation_12.pde_2d_wave_fourier_sol import PDE2DWaveFourierSolution

f_xyt_0 = lambda x, y: x * (x - 1) * y * (y - 1)  # 初值条件
df_xyt_0 = lambda x, y: 0.0 * x * y  # 初值条件
c, a, b, t_T = 1 / np.pi, 1, 1, 0
m, n = 5, 5

heat_fourier = PDE2DWaveFourierSolution(f_xyt_0, df_xyt_0, c, a, b, t_T, m, n)
uxyt = heat_fourier.solve_pde()

# 如下可视化6个时刻的数值解曲面
xi, yi = np.linspace(0, a, 50), np.linspace(0, b, 50)
x_, y_ = np.meshgrid(xi, yi)
T = np.linspace(0, 5, 6)  # 定义6个时刻
fig = plt.figure(figsize=(18, 10))
for i, t in enumerate(T):
    ax = fig.add_subplot(231 + i, projection='3d')
    num_sol = uxyt(x_, y_, t)  # 数值解
    ax.plot_surface(x_, y_, num_sol, cmap='rainbow')
    ax.set_xlabel("$x$", fontdict={"fontsize": 18})
    ax.set_ylabel("$y$", fontdict={"fontsize": 18})
    # ax.set_zlabel("$U$", fontdict={"fontsize": 18})
    # z_format = plt.FormatStrFormatter('%.e')  # 设置y轴标签文本的格式
    # ax.zaxis.set_major_formatter(z_format)
    plt.title("二维波动方程的傅里叶解 $t = %d$" % t, fontdict={"fontsize": 18})
    plt.tick_params(labelsize=16)  # 刻度字体大小16
plt.show()
