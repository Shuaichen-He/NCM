# -*- coding: UTF-8 -*-
"""
@file_name: test_cubic_and_b_spline.py
@time: 2022-11-02
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import sympy
from interpolation_02.cubic_spline_interpolation import CubicSplineInterpolation
from interpolation_02.b_spline_interpolation import BSplineInterpolation
import matplotlib.pyplot as plt

# x = np.linspace(-4 * np.pi, 4 * np.pi, 20, endpoint=True)
# fun_expr = lambda x: 50 * np.sin(x) / x
# y = fun_expr(x)  # 函数值
# dy = 50 * np.cos(x) / x - 50 * np.sin(x) / x ** 2  # 一阶导数值
# d2y = 50 * (-x ** 2 * np.sin(x) - 2 * x * np.cos(x) + 2 * np.sin(x)) / x ** 3  # 二阶导数值
# x0 = np.array([-3.48, -1.69, 0.05, 2.66, 4.08, 4.876])
#
# plt.figure(figsize=(14, 5))
# csi = CubicSplineInterpolation(x, y, dy=dy, d2y=d2y, boundary_cond="second")
# csi.fit_interp()
# y0 = csi.predict_x0(x0)
# print(csi.poly_coefficient)
# plt.subplot(121)
# csi.plt_interpolation(fh=fun_expr, is_show=False)
# print("=" * 100)
#
# bsi = BSplineInterpolation(x, y, dy=dy, d2y=d2y, boundary_cond="second")
# bsi.fit_interp()
#
# y0 = bsi.predict_x0(x0)
# print(bsi.poly_coefficient)
# plt.subplot(122)
# bsi.plt_interpolation(fh=fun_expr, is_show=False)
# plt.show()


# fun_expr = lambda x: 11 * np.sin(x) + 7 * np.cos(5 * x)
# x = np.linspace(-1, 3, 25, endpoint=True)
# print(x)
# y = fun_expr(x)  # 取值模拟
# x0 = np.asarray([-0.9, -0.2, 1.5, 2.2, 2.7, 2.9])
# dy = 11 * np.cos(x) - 35 * np.sin(5 * x)
# d2y = -11 * np.sin(x) - 175 * np.cos(5 * x)

t = sympy.Symbol("t")
x = np.linspace(-np.pi, np.pi, 19, endpoint=True)
fun_sym = sympy.tan(sympy.cos((sympy.sqrt(3) + sympy.sin(2 * t)) / (3 + 4 * t ** 2)))
fun_expr = sympy.lambdify(t, fun_sym)
y = fun_expr(x)  # 函数值
df_expr = sympy.lambdify(t, fun_sym.diff(t, 1))
dy = df_expr(x)  # 一阶导数值
d2f_expr = sympy.lambdify(t, fun_sym.diff(t, 2))
d2y = d2f_expr(x)  # 二阶导数值

a, b = -np.pi, np.pi
plt.figure(figsize=(14, 5))
plt.subplot(121)
bsi = BSplineInterpolation(x, y, dy=dy, d2y=d2y, boundary_cond="second")
bsi.fit_interp()
bsi.plt_interpolation(fh=fun_expr, is_show=False)
plt.subplot(122)
xi = np.linspace(a, b, 200, endpoint=True)
boundary_cond = ["complete", "second", "natural", "periodic"]
ls_ = ["-", "--", "-.", ":"]
error = np.zeros((200, 4))
for i, bc in enumerate(boundary_cond):
    bsi = BSplineInterpolation(x, y, dy=dy, d2y=d2y, boundary_cond=bc)
    bsi.fit_interp()
    # print(bsi.poly_coefficient)
    # print("=" * 80)
    yi = bsi.predict_x0(xi)
    mse = np.mean((yi - fun_expr(xi)) ** 2)
    print(mse)
    error[:, i] = np.abs(yi - fun_expr(xi))
    plt.semilogy(xi[1:-1], error[1:-1, i], ls_[i], lw=2, label="$%s: \ MSE=%.5e$" % (bc, mse))
plt.xlabel("$x$", fontdict={"fontsize": 18})  # 阶次
plt.ylabel(r"$\vert f(x_k) - g(x_k) \vert$", fontdict={"fontsize": 18})  # 精度
plt.title("三次均匀B样条插值不同边界条件下的绝对值误差", fontdict={"fontsize": 18})
plt.legend(frameon=False, fontsize=16)
plt.tick_params(labelsize=18)  # 刻度字体大小18
plt.grid(ls=":")
plt.ylim([1e-9, 1e-2])
plt.show()

# plt.figure(figsize=(14, 9))
# boundary_cond = ["complete", "second"]
# for i, bc in enumerate(boundary_cond):
#     csi = CubicSplineInterpolation(x, y, dy=dy, d2y=d2y, boundary_cond=bc)
#     csi.fit_interp()
#     print(csi.poly_coefficient)
#     # y0 = csi.predict_x0(x0)
#     # print(bc + "插值点的值：", y0, "\n误差：", fun_expr(x0) - y0)
#     plt.subplot(221 + i)
#     csi.plt_interpolation(fh=fun_expr, is_show=False)
#
# for i, bc in enumerate(boundary_cond):
#     bsi = BSplineInterpolation(x, y, dy=dy, d2y=d2y, boundary_cond=bc)
#     bsi.fit_interp()
#     # y0 = bsi.predict_x0(x0)
#     # print(bc + "插值点的值：", y0, "\n误差：", fun_expr(x0) - y0)
#     plt.subplot(223 + i)
#     bsi.plt_interpolation(fh=fun_expr, is_show=False)
# plt.show()