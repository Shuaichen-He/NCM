# -*- coding: UTF-8 -*-
"""
@author:Lenovo
@file:test_cubic_bspline.py
@time:2021/08/30
"""
import numpy as np
import sympy
from interpolation_02.b_spline_interpolation import BSplineInterpolation
from interpolation_02.lagrange_interpolation import LagrangeInterpolation
import matplotlib.pyplot as plt

# 正弦周期函数
# x = np.linspace(-2 * np.pi, 2 * np.pi, 20, endpoint=True)
# fun_expr = lambda x: np.sin(x)  # 正弦函数模拟
# y = fun_expr(x)
# dy = np.cos(x)
# d2y = -np.sin(x)
# x0 = np.array([np.pi / 2, 2.6, 2 * np.pi, 4.0, 4.8])
#
# plt.figure(figsize=(14, 5))
# boundary_cond = ["natural", "periodic"]
# for i, bc in enumerate(boundary_cond):
#     csi = CubicSplineInterpolation(x, y, dy=dy, d2y=d2y, boundary_cond=bc)
#     csi.fit_interp()
#     y0 = csi.predict_x0(x0)
#     print(bc + "插值点的值：", y0, "\n误差：", np.array(fun_expr(x0) - y0))
#     plt.subplot(121 + i)
#     csi.plt_interpolation(fh=fun_expr, is_show=False)
# plt.show()

a, b = -4 * np.pi, 4 * np.pi
t = sympy.Symbol("t")
x = np.linspace(a, b, 20, endpoint=True)
fun_sym = 50 * sympy.sin(t) / t
# fun_sym = sympy.sin(t)
fun_expr = sympy.lambdify(t, fun_sym)
y = fun_expr(x)  # 函数值
df_expr = sympy.lambdify(t, fun_sym.diff(t, 1))
dy = df_expr(x)  # 一阶导数值
d2f_expr = sympy.lambdify(t, fun_sym.diff(t, 2))
d2y = d2f_expr(x)  # 二阶导数值
x0 = np.array([-3.48, -1.69, 0.05, 2.66, 4.08, 4.876])

# fun_expr = lambda x: 11 * np.sin(x) + 7 * np.cos(5 * x)
# x = np.linspace(-1, 3, 20, endpoint=True)
# y = fun_expr(x)  # 取值模拟
# x0 = np.asarray([-0.9, -0.2, 1.5, 2.2, 2.7, 2.9])
# dy = 11 * np.cos(x) - 35 * np.sin(5 * x)
# d2y = -11 * np.sin(x) - 175 * np.cos(5 * x)
# print("精确值：", fun_expr(x0))

plt.figure(figsize=(14, 5))
plt.subplot(121)
csi = BSplineInterpolation(x, y, dy=dy, d2y=d2y, boundary_cond="complete")
csi.fit_interp()
y0 = csi.predict_x0(x0)
print("complete插值点的值：", y0, "\n误差：", np.array(fun_expr(x0) - y0))
# print(csi.poly_coefficient)
csi.plt_interpolation(fh=fun_expr, is_show=False)
plt.subplot(122)
xi = np.linspace(a, b, 200, endpoint=True)
boundary_cond = ["complete", "second", "natural", "periodic"]
ls_ = ["-", "--", "-.", ":"]
error = np.zeros((200, 4))
for i, bc in enumerate(boundary_cond):
    csi = BSplineInterpolation(x, y, dy=dy, d2y=d2y, boundary_cond=bc)
    csi.fit_interp()
    y0 = csi.predict_x0(x0)
    print(bc + "插值点的值：", y0, "\n误差：", np.array(fun_expr(x0) - y0))
    # print(csi.poly_coefficient)
    yi = csi.predict_x0(xi)
    mse = np.mean((yi - fun_expr(xi)) ** 2)
    error[:, i] = np.abs(yi - fun_expr(xi))
    plt.semilogy(xi[1:-1], error[1:-1, i], ls_[i], lw=2, label="$%s: \ MSE=%.2e$" % (bc, mse))
plt.xlabel("$x$", fontdict={"fontsize": 18})  # 阶次
plt.ylabel(r"$\vert f(x_k) - g(x_k) \vert$", fontdict={"fontsize": 18})  # 精度
plt.title("三次样条插值不同边界条件下的绝对值误差", fontdict={"fontsize": 18})
plt.legend(frameon=False, fontsize=16)
plt.tick_params(labelsize=18)  # 刻度字体大小18
plt.grid(ls=":")

# plt.ylim([2e-7, 1e-3])  # 正弦函数
plt.show()

# 例4
# x = np.array([0.25, 0.30, 0.39, 0.45, 0.53])
# y = np.array([0.500, 0.5477, 0.6245, 0.6708, 0.7280])
# dy = np.array([1.0000, 0.6868])
# d2y = np.array([0, 0])
# boundary_cond = ["complete", "natural"]
# plt.figure(figsize=(14, 5))
# for i, bc in enumerate(boundary_cond):
#     csi = CubicSplineInterpolation(x, y, dy=dy, d2y=d2y, boundary_cond=bc)
#     csi.fit_interp()
#     print(csi.poly_coefficient)
#     plt.subplot(121 + i)
#     csi.plt_interpolation(is_show=False)
# plt.show()

# 例5示例
# x = np.linspace(0, 24, 13, endpoint=True)  # 时间
# y = np.array([12, 9, 9, 10, 18, 24, 28, 27, 25, 20, 18, 15, 13])  # 维度值
# x0 = [13, 17.4, 22.8]
#
# plt.figure(figsize=(14, 5))
# plt.subplot(121)
# csi = CubicSplineInterpolation(x, y, boundary_cond="natural")
# csi.fit_interp()
# y0 = csi.predict_x0(x0)
# print("自然边界条件求解插值点的值：", y0)
# csi.plt_interpolation(x0, y0, is_show=False)
# plt.subplot(122)
# lag = LagrangeInterpolation(x, y)
# lag.fit_interp()
# y0 = lag.predict_x0(x0)
# print("插值：", y0)
# lag.plt_interpolation(x0, y0, is_show=False)
# plt.show()