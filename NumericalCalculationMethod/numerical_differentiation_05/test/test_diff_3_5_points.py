# -*- coding: UTF-8 -*-
"""
@file:test_diff_3_5_points.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import sympy
import matplotlib.pyplot as plt
from numerical_differentiation_05.three_five_points_formula_differentiation \
    import ThreeFivePointsFormulaDifferentiation

x = sympy.Symbol("x")
fun = sympy.sin(x) * sympy.exp(-x)
x0 = np.array([1.23, 1.75, 1.89, 2.14, 2.56])
tfpfd = ThreeFivePointsFormulaDifferentiation(fun, h=0.05, points_type="five", diff_type="middle")
y0 = tfpfd.predict_diff_x0(x0)
plt.figure(figsize=(14, 5))
plt.subplot(121)
tfpfd.plt_differentiation([1, 3], x0, y0, is_show=False, is_fh_marker=True)
diff_val = sympy.lambdify(x, fun.diff(x, 1))(x0)
print("微分值：", y0, "\n精确值：", diff_val, "\n误差：", y0 - diff_val)

diff5_types = ["first", "second", "middle", "four", "five"]
np.random.seed(5)
# xi = 1 + np.random.rand(100) * 4  # [1, 5]区间随机100个点
# xi = np.asarray(sorted(xi))
xi = np.linspace(1, 3, 200)
y_true = sympy.lambdify(x, fun.diff(x, 1))(xi)  # 精确微分值
diff_error = np.zeros((len(xi), 5))  # 存储每种类型的误差向量
mae = []  # 存储平均绝对值误差
for i, dt in enumerate(diff5_types):
    tfpfd = ThreeFivePointsFormulaDifferentiation(fun, h=0.05, points_type="five", diff_type=dt)
    diff_error[:, i] = y_true - tfpfd.predict_diff_x0(xi)
    mae.append(np.mean(np.abs(diff_error[:, i])))
plt.subplot(122)  # 略去图形修饰代码，主要包括坐标轴名称、刻度大小、图例和标题信息等...
line_style = ["--", "-.", "-", ":", "--"]
for i in range(5):
    plt.semilogy(xi, np.abs(diff_error[:, i]), line_style[i], lw=2,
                 label="$%s$" % diff5_types[i] + "$: MAE=%.2e$" % mae[i])
plt.xlabel("$x$", fontdict={"fontsize": 18})
plt.ylabel(r"$err=\vert f^{\prime}(x) - \hat f^{\prime}(x) \vert$", fontdict={"fontsize": 18})
plt.title("不同的五点公式法数值微分精度比较$h=0.05$", fontdict={"fontsize": 18})
plt.legend(frameon=False, fontsize=16)
plt.tick_params(labelsize=16)  # 刻度字体大小16
plt.grid(ls=":")
plt.ylim([1e-8, 1e-4])
plt.show()



# x = sympy.Symbol("x")
# fun = x ** 2 * sympy.exp(-x)
# x0 = np.array([1.23, 1.75, 1.89, 2.14, 2.56])
# tfpfd = ThreeFivePointsFormulaDifferentiation(fun, h=0.05, points_type="five", diff_type="middle")
# y0 = tfpfd.predict_diff_x0(x0)
# plt.figure(figsize=(14, 5))
# plt.subplot(121)
# tfpfd.plt_differentiation([1, 3], x0, y0, is_show=False, is_fh_marker=True)
# diff_val = sympy.lambdify(x, fun.diff(x, 1))(x0)
# print("微分值：", y0, "\n精确值：", diff_val, "\n误差：", y0 - diff_val)
#
# diff5_types = ["first", "second", "middle", "four", "five"]
# xi = 1 + np.random.rand(100) * 4  # [1, 5]区间随机100个点
# xi = np.asarray(sorted(xi))
# y_true = sympy.lambdify(x, fun.diff(x, 1))(xi)  # 精确微分值
# diff_error = np.zeros((len(xi), 5))  # 存储每种类型的误差向量
# mse = []  # 存储均方根误差
# for i, dt in enumerate(diff5_types):
#     tfpfd = ThreeFivePointsFormulaDifferentiation(fun, h=0.05, points_type="five", diff_type=dt)
#     diff_error[:, i] = y_true - tfpfd.predict_diff_x0(xi)
#     mse.append(np.sqrt(np.mean(diff_error[:, i] ** 2)))
#
# plt.subplot(122)
# line_style = ["--o", "-.d", "-*", ":s", "--p"]
# for i in range(5):
#     plt.plot(xi, diff_error[:, 0], line_style[i], markersize=5, label="$%s$" % diff5_types[i] + "$: %.2e$" % mse[i])
# plt.xlabel("$x$(随机100个点)", fontdict={"fontsize": 18})
# plt.ylabel("微分值误差", fontdict={"fontsize": 18})
# plt.title("不同的五点公式法数值微分精度比较$h=0.05$", fontdict={"fontsize": 18})
# plt.legend(frameon=False, fontsize=16)
# plt.tick_params(labelsize=16)  # 刻度字体大小16
# plt.grid(ls=":")
# plt.show()
