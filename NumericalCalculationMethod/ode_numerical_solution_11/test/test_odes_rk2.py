# -*- coding: UTF-8 -*-
"""
@file_name: test_odes_rk.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import matplotlib.pyplot as plt
from ode_numerical_solution_11.first_order_ODEs_RK import FirstOrderODEsRK

def ode_funs(x, y):
    k1, k2 = 1, 1000
    return np.array([-k1 * y[0], k1 * y[0] - k2 * y[1], -k2 * y[1]])

x0, y0, h = 0, np.array([1, 1, 1]), 0.0001  # 参数
odes_rk = FirstOrderODEsRK(ode_funs, x0, y0, x_final=10, h=h)
odes_rk.fit_odes()
plt.figure(figsize=(14, 5))
plt.subplot(121)
odes_rk.plt_odes_rk(is_show=False)  # 不存在解析解，则直接绘制数值解即可
plt.subplot(122)
odes_rk = FirstOrderODEsRK(ode_funs, x0, y0, x_final=50, h=h)
odes_rk.fit_odes()
odes_rk.plt_odes_rk(is_show=False)  # 不存在解析解，则直接绘制数值解即可
plt.show()