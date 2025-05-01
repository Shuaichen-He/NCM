# -*- coding: UTF-8 -*-
"""
@file_name: test_richason_1.py
@time: 2021-11-24
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import matplotlib.pyplot as plt
from numerical_differentiation_05.richardson_extrapolation_differentiation \
    import RichardsonExtrapolationDifferentiation

fun = lambda x: x ** 2 * np.exp(-x)  # 微分函数

dfun = lambda x: np.exp(-x) * (2 * x - x ** 2)  # 一阶导函数

x0 = np.linspace(0, 11, 9)
print(x0)

y_true = dfun(x0)
plt.figure(figsize=(14, 5))
plt.subplot(121)
red = RichardsonExtrapolationDifferentiation(fun, step=9)
diff_value = red.predict_diff_x0(x0)
red.plt_differentiation([0, 11], dfun, x0, diff_value, is_show=False, is_fh_marker=True)
plt.subplot(122)
step_vector = [3, 5, 7, 9]
xi = np.linspace(0, 11, 200)
ls_ = ["--", "-.", ":", "-"]
for i, step in enumerate(step_vector):
    red = RichardsonExtrapolationDifferentiation(fun, step=step)
    yi = red.predict_diff_x0(xi)
    error = np.abs(yi - dfun(xi))
    mae = np.mean(error)
    print("最大绝对值误差：%.10e" % np.max(error))
    print("平均绝对值误差：%.10e" % mae)
    plt.semilogy(xi, error, ls_[i], lw=2, label="$step=%d$" % step)
    diff_value = red.predict_diff_x0(x0)
    print("微分值：", diff_value, "\n误差：", diff_value - y_true)
    print("=" * 80)
plt.legend(frameon=False, fontsize=18, ncol=2)
plt.xlabel(r"$x$", fontdict={"fontsize": 18})
plt.ylabel(r"$err = \vert f^{\prime}(x)- \hat f^{\prime}(x) \vert$", fontdict={"fontsize": 18})
plt.title("理查森外推算法数值微分误差曲线", fontdict={"fontsize": 18})
plt.tick_params(labelsize=16)  # 刻度字体大小16
plt.grid(ls=":")
plt.show()
