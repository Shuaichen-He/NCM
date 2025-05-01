# -*- coding: UTF-8 -*-
"""
@author:Lenovo
@file:test_b_spline.py
@time:2021/08/30
"""
import numpy as np
import sympy
from interpolation_02.b_spline_interpolation import BSplineInterpolation
import matplotlib.pyplot as plt

# x = np.linspace(-2 * np.pi, 2 * np.pi, 20, endpoint=True)
# fun_expr = lambda x: np.sin(x)  # 正弦函数模拟
# y = fun_expr(x)
# dy = np.cos(x)
# d2y = -np.sin(x)
# x0 = np.array([np.pi / 2, 2.6, 2 * np.pi, 4.0, 4.8])

t = sympy.Symbol("t")
x = np.linspace(-4 * np.pi, 4 * np.pi, 20, endpoint=True)
fun_sym = 50 * sympy.sin(t) / t
fun_expr = sympy.lambdify(t, fun_sym)
y = fun_expr(x)  # 函数值
df_expr = sympy.lambdify(t, fun_sym.diff(t, 1))
dy = df_expr(x)  # 一阶导数值
d2f_expr = sympy.lambdify(t, fun_sym.diff(t, 2))
d2y = d2f_expr(x)  # 二阶导数值
x0 = np.array([x[1], x[3], 2.8, 3.5])

# fun_expr = lambda x: 11 * np.sin(x) + 7 * np.cos(5 * x)
# x = np.linspace(-1, 3, 20, endpoint=True)
# y = fun_expr(x)  # 取值模拟
# x0 = np.asarray([-0.9, -0.2, 1.5, 2.2, 2.7, 2.9])
# dy = 11 * np.cos(x) - 35 * np.sin(5 * x)
# d2y = -11 * np.sin(x) - 175 * np.cos(5 * x)
# print("精确值：", fun_expr(x0))

plt.figure(figsize=(14, 9))
boundary_cond = ["complete", "second", "natural", "periodic"]
# boundary_cond = ["natural", "periodic"]
for i, bc in enumerate(boundary_cond):
    bsi = BSplineInterpolation(x, y, dy=dy, d2y=d2y, boundary_cond=bc)
    bsi.fit_interp()
    y0 = bsi.predict_x0(x0)
    print(bc + "插值点的值：", y0, "\n误差：", fun_expr(x0) - y0)
    # print(bsi.poly_coefficient)
    plt.subplot(221 + i)
    bsi.plt_interpolation(fh=fun_expr, is_show=False)
plt.show()

# time = np.linspace(0, 24, 25)  # 时间
# speed = np.array([0, .45, 1.79, 4.02, 7.15, 11.18, 16.09, 21.90, 29.05, 29.05,
#                   29.05, 29.05, 29.05, 22.42, 17.9, 17.9, 17.9, 17.9, 14.34,
#                   11.01, 8.9, 6.54, 2.03, 0.55, 0])  # 速度数据
# bsi = BSplineInterpolation(time, speed, boundary_cond="natural")
# bsi.fit_interp()
# plt.figure(figsize=(8, 6))
# plt.plot(time,speed, "k--", label="base point")
# bsi.plt_interpolation(is_show=False)
# plt.show()
