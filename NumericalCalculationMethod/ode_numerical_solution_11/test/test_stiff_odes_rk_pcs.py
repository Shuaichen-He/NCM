# -*- coding: UTF-8 -*-
"""
@file_name: test_stiff_odes_rk.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import time
import matplotlib.pyplot as plt
from ode_numerical_solution_11.stiff_ODES_rk_pcs import StiffODEsRKPCS

def ode_funs(x, y):
    dy = np.array([-0.013 * y[0] - 1000 * y[0] * y[1], -2500 * y[1] * y[2],
                   -0.013 * y[0] - 1000 * y[0] * y[1] - 2500 * y[1] * y[2]])
    return dy


def ode_funs2(x, y):
    k1, k2 = 1, 1000  # 此处修改k2值即可
    return np.array([-k1 * y[0], k1 * y[0] - k2 * y[1], -k2 * y[1]])


def lorenz_ode(t, y):
    sigma, rou, beta = 16, 45, 4
    return np.array([sigma * (y[1] - y[0]), y[0] * (rou - y[2]) - y[1], y[0] * y[1] - beta * y[2]])


# 例9测试代码
x0, y0, h = 0, np.array([1, 1, 0]), 0.0001
plt.figure(figsize=(14, 5))
plt.subplot(121)
stiff = StiffODEsRKPCS(ode_funs, x0, y0, x_final=1, h=h)
stiff.fit_odes()
stiff.plt_odes_rk(is_show=False)
plt.subplot(122)
stiff = FirstOrderODEsRK(ode_funs, x0, y0, x_final=0.025, h=h)
stiff.fit_odes()
stiff.plt_odes_rk(is_show=False)
plt.show()

# 例10测试代码
# x0, y0, h = 0, np.array([1, 1, 1]), 0.0001  # 方程2
# odes_rk = StiffODEsRKPCS(ode_funs2, x0, y0, x_final=20, h=h)
# time_ = 0.0
# for i in range(20):
#     start = time.time()
#     odes_rk.fit_odes()
#     end = time.time()
#     time_ += end - start
# print(time_ / 20, odes_rk.ode_sol[-1, 1:])
# odes_rk.plt_odes_rk()

# plt.figure(figsize=(14, 5))
# plt.subplot(121)
# x0, y0, h = 0, np.array([1, 1, 1]), 0.0001  # 方程2
# odes_rk = StiffODEsRKPCS(ode_funs2, x0, y0, x_final=20, h=h)
# odes_rk.fit_odes()
# odes_rk.plt_odes_rk(is_show=False)
# plt.subplot(122)
# odes_rk = StiffODEsRKPCS(ode_funs2, x0, y0, x_final=50, h=h)
# odes_rk.fit_odes()
# odes_rk.plt_odes_rk(is_show=False)
# plt.show()

# 例11测试代码
x0, y0, h = 0, np.array([12, 4, 1]), 0.001  # 方程参数设置
stiff = StiffODEsRKPCS(lorenz_ode, x0, y0, x_final=40, h=h)
stiff.fit_odes()
# 绘制洛伦兹方程曲线
fig = plt.figure(figsize=(8, 6))
ax = fig.gca(projection='3d')
ax.plot(stiff.ode_sol[:, 1], stiff.ode_sol[:, 2], stiff.ode_sol[:, 3])
ax.set_xlabel("$x$", fontdict={"fontsize": 18})
ax.set_ylabel("$y$", fontdict={"fontsize": 18})
ax.set_zlabel("$z$", fontdict={"fontsize": 18})
plt.title("$RKPCS$法求解洛伦兹方程$[0, 40]$", fontdict={"fontsize": 18})
plt.tick_params(labelsize=18)  # 刻度字体大小16
plt.show()
