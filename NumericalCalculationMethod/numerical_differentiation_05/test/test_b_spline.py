# -*- coding: UTF-8 -*-
"""
@author:Lenovo
@file:test_cubic_spline.py
@time:2021/08/26
"""
import numpy as np
import matplotlib.pyplot as plt
from numerical_differentiation_05.cubic_bspline_differentiation import CubicBSplineDifferentiation

# def fun(x):
#     return np.sin(x) * np.exp(-x)


diff_fun = lambda x: np.log(x) * np.sin(x)  # 微分函数

dfun = lambda x: np.sin(x) / x + np.log(x) * np.cos(x)  # 一阶导函数

# x0 = np.array([1.23, 1.75, 1.89, 2.14, 2.56])
x0 = np.linspace(2, 10, 9)

plt.figure(figsize=(14, 5))
plt.subplot(121)
cbsd = CubicBSplineDifferentiation(diff_fun, n=9, h=0.1)
y0 = cbsd.predict_diff_x0(x0)
cbsd.plt_differentiation([2, 11], dfun, x0, y0, is_show=False, is_fh_marker=True)
plt.subplot(122)
n_num = [3, 5, 7, 9]
xi = np.linspace(2, 11, 200)
ls_ = ["--", "-.", ":", "-"]
for i, n in enumerate(n_num):
    cbsd = CubicBSplineDifferentiation(diff_fun, n=n, h=0.1)
    yi = cbsd.predict_diff_x0(xi)
    error = np.abs(yi - dfun(xi))
    mae = np.mean(error)
    print("最大绝对值误差：%.10e" % np.max(error))
    print("平均绝对值误差：%.10e" % mae)
    plt.semilogy(xi, error, ls_[i], lw=2, label="$n=%d$" % (n))
    print("微分值：", y0, "\n误差：", cbsd.predict_diff_x0(x0) - dfun(x0))
    print("=" * 80)
plt.legend(frameon=False, fontsize=18, ncol=2)
plt.xlabel(r"$x$", fontdict={"fontsize": 18})
plt.ylabel(r"$err = \vert f^{\prime}(x)- \hat f^{\prime}(x) \vert$", fontdict={"fontsize": 18})
plt.title("三次均匀$B$样条插值数值微分误差曲线", fontdict={"fontsize": 18})
plt.tick_params(labelsize=16)  # 刻度字体大小16
plt.grid(ls=":")
plt.ylim([1e-7, 1e-1])
plt.show()
# diff_val = np.exp(-x0) * (np.cos(x0) - np.sin(x0))
# diff_val = np.sin(x0) / x0 + np.log(x0) * np.cos(x0)
# print("微分值：", y0, "\n精确值：", diff_val, "\n误差：", y0 - diff_val)
