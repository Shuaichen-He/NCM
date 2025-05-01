# -*- coding: UTF-8 -*-
"""
@author:Lenovo
@file:test_cubic_spline.py
@time:2021/08/26
"""
import numpy as np
import matplotlib.pyplot as plt
from numerical_differentiation_05.adaptive_cubic_bspline_differentiation \
    import AdaptiveCubicBSplineDifferentiation

fun = lambda x: np.log(x) * np.sin(x)  # 微分函数

dfun = lambda x: np.sin(x) / x + np.log(x) * np.cos(x)  # 一阶导函数

# x0 = np.array([1.23, 1.75, 1.89, 2.14, 2.56])
x0 = np.linspace(2, 10, 9)
plt.figure(figsize=(14, 5))
plt.subplot(121)
acbsd = AdaptiveCubicBSplineDifferentiation(fun, h=0.1, eps=1e-9)
y0 = acbsd.cal_diff(x0)
acbsd.plt_differentiation([2, 11], dfun, x0, y0, is_show=False, is_fh_marker=True)

plt.subplot(122)
xi = np.linspace(2, 11, 200)
ls_ = ["--", "-.", ":", "-"]
eps_vector = [1e-3, 1e-5, 1e-7, 1e-9]
for i, eps in enumerate(eps_vector):
    acbsd = AdaptiveCubicBSplineDifferentiation(fun, h=0.1, eps=eps)
    yi = acbsd.cal_diff(xi)
    error = np.abs(yi - dfun(xi))
    mae = np.mean(error)
    print("最大绝对值误差：%.10e" % np.max(error))
    print("平均绝对值误差：%.10e" % mae)
    plt.semilogy(xi, error, ls_[i], lw=2, label="$\epsilon=%.2e$" % (eps))
    y0 = acbsd.cal_diff(x0)
    print("微分值：", y0, "\n误差：", y0 - dfun(x0))
    print("=" * 80)
plt.legend(frameon=False, fontsize=18, ncol=2)
plt.xlabel(r"$x$", fontdict={"fontsize": 18})
plt.ylabel(r"$err = \vert f^{\prime}(x)- \hat f^{\prime}(x) \vert$", fontdict={"fontsize": 18})
plt.title("自适应三次均匀$B$样条插值数值微分误差曲线", fontdict={"fontsize": 18})
plt.tick_params(labelsize=16)  # 刻度字体大小16
plt.grid(ls=":")
# plt.ylim([1e-7, 1e-1])
plt.show()