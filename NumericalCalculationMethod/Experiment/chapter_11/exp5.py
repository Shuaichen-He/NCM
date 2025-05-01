# -*- coding: UTF-8 -*-
"""
@file_name: exp5.py
@time: 2023-02-11
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import matplotlib.pyplot as plt
from ode_numerical_solution_11.stiff_ODES_rk_pcs import StiffODEsRKPCS
from ode_numerical_solution_11.first_order_ODEs_RK import FirstOrderODEsRK


def ode_funs(x, y):
    """
    微分方程组定义
    :param x: 自变量
    :param y: y[0]代表y1, y[1]代表y2, y[2]代表y3
    :return:
    """
    dy = np.array([-0.013 * y[0] - 1000 * y[0] * y[1],
                   -2500 * y[1] * y[2],
                   -0.013 * y[0] - 1000 * y[0] * y[1] - 2500 * y[1] * y[2]])
    return dy


x0, y0, h, x_final= 0, np.array([1, 1, 0]), 0.0001, 0.025

plt.figure(figsize=(14, 5))
# 龙格—库塔法
frk = FirstOrderODEsRK(ode_funs, x0, y0, x_final=x_final, h=h)
frk.fit_odes()
plt.subplot(121)
frk.plt_odes_rk(is_show=False)
print(frk.ode_sol[-1, :])

# 3 阶显式龙格 — 库塔公式 + 3 级 6 阶隐式龙格 — 库塔公式构成预测校正系统
s_ode = StiffODEsRKPCS(ode_funs, x0, y0, x_final=x_final, h=h)
s_ode.fit_odes()
plt.subplot(122)
s_ode.plt_odes_rk_pcs(is_show=False)
plt.show()
print(s_ode.ode_sol[-1, :])
