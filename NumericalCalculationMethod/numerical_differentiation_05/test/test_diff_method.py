# -*- coding: UTF-8 -*-
"""
@author:Lenovo
@file:test_diff_method.py
@time:2021/08/26
"""
import numpy as np
import sympy
import matplotlib.pyplot as plt
import time
from numerical_differentiation_05.richardson_extrapolation_differentiation \
    import RichardsonExtrapolationDifferentiation
from numerical_differentiation_05.cubic_bspline_differentiation import CubicBSplineDifferentiation
from numerical_differentiation_05.adaptive_cubic_bspline_differentiation \
    import AdaptiveCubicBSplineDifferentiation
from numerical_differentiation_05.three_five_points_formula_differentiation \
    import ThreeFivePointsFormulaDifferentiation


fun = lambda x: x ** 2 * np.exp(-x) # 微分函数
dfun = lambda x: np.exp(-x) * (2 * x - x ** 2) # 一阶导函数

x0, xi = np.linspace(0, 5, 9), np.linspace(0, 11, 200)
y_true = dfun(x0)
d_err = np.zeros((len(xi), 4))  # 各微分误差
diff_value = np.zeros(len(x0))  # 给定点的微分值

time_consumption = []
red = RichardsonExtrapolationDifferentiation(fun, step=6)
d_err[:, 0] = red.predict_diff_x0(xi)
for i in range(100):
    start = time.time()
    diff_value = red.predict_diff_x0(x0)
    end = time.time()
    time_consumption.append(end - start)
print("平均时间消耗：", np.mean(time_consumption))
print("微分值：", diff_value, "\n精确值：", y_true, "\n误差：", diff_value - y_true)
print("-" * 80)

time_consumption = []
cbsd = CubicBSplineDifferentiation(fun, n=50, h=0.05)
d_err[:, 1] = cbsd.predict_diff_x0(xi)
for i in range(100):
    start = time.time()
    diff_value = cbsd.predict_diff_x0(x0)
    end = time.time()
    time_consumption.append(end - start)
print("平均时间消耗：", np.mean(time_consumption))
print("微分值：", diff_value, "\n误差：", diff_value - y_true)
print("-" * 80)

time_consumption = []
acsd = AdaptiveCubicBSplineDifferentiation(fun, h=0.1, eps=1e-7)
d_err[:, 2] = acsd.cal_diff(xi)
for i in range(100):
    start = time.time()
    diff_value = acsd.cal_diff(x0)
    end = time.time()
    time_consumption.append(end - start)
print("平均时间消耗：", np.mean(time_consumption))
print("微分值：", diff_value, "\n误差：", diff_value - y_true)
print("-" * 80)

time_consumption = []
x = sympy.Symbol("x")
fun = x ** 2 * sympy.exp(-x)
tfpffd = ThreeFivePointsFormulaDifferentiation(fun, points_type="five", diff_type="middle", h=0.02)
d_err[:, 3] = tfpffd.predict_diff_x0(xi)
for i in range(100):
    start = time.time()
    diff_value = tfpffd.predict_diff_x0(x0)
    end = time.time()
    time_consumption.append(end - start)
print("平均时间消耗：", np.mean(time_consumption))
print("微分值：", diff_value, "\n误差：", diff_value - y_true)
print("-" * 80)

plt.figure(figsize=(7, 5))
ls_ = ["--", "-.", ":", "-"]
labels = ["Richardson", "BSpline", "AdaBSpline", "Five(middle)"]
for i in range(4):
    error = np.abs(d_err[:, i] - dfun(xi))
    mae = np.mean(error)
    print("最大绝对值误差：%.10e" % np.max(error))
    print("平均绝对值误差：%.10e" % mae)
    plt.semilogy(xi, error, ls_[i], lw=2, label="$%s$" % (labels[i]))
plt.legend(frameon=False, fontsize=16)
plt.xlabel(r"$x$", fontdict={"fontsize": 18})
plt.ylabel(r"$err = \vert f^{\prime}(x)- \hat f^{\prime}(x) \vert$", fontdict={"fontsize": 18})
plt.title("各数值微分方法求解一阶数值微分误差曲线", fontdict={"fontsize": 18})
plt.tick_params(labelsize=16)  # 刻度字体大小16
plt.grid(ls=":")
plt.show()
