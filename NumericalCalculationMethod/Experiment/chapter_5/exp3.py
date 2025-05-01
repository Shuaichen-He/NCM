# -*- coding: UTF-8 -*-
"""
@file_name: exp3.py
@time: 2023-02-10
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import sympy
import matplotlib.pyplot as plt
from numerical_differentiation_05.richardson_extrapolation_differentiation \
    import RichardsonExtrapolationDifferentiation
from numerical_differentiation_05.five_points_second_order_differentiation \
    import FivePointsSecondOrderDifferentiation
from numerical_differentiation_05.cubic_bspline_2_order_differentiation import CubicBSpline2OrderDifferentiation

fun = lambda x: x ** 2 + x ** (1/3) + np.sin(x + np.cos(x) ** 2)  # 微分函数
x = sympy.symbols("x")
fun_sym = x ** 2 + x ** (1/3) + sympy.sin(x + sympy.cos(x) ** 2)  # 微分函数，符号定义
dfun = sympy.lambdify(x, fun_sym.diff())  # 一阶导函数

np.random.seed(100)
x0 = 1 + 4 * np.random.rand(10)  # 随机生成，均匀分布
x0 = np.asarray(sorted(x0))  # 排序
y_true = dfun(x0)  # 一阶微分真值
print("真值x：", x0)
print("真值dy：", y_true)

# (1) 理查森外推算法
plt.figure(figsize=(14, 5))
for i, step in enumerate([3, 9]):
    plt.subplot(121 + i)
    red = RichardsonExtrapolationDifferentiation(fun, step=step)
    diff_value = red.predict_diff_x0(x0)
    print("近似微分值：", diff_value)
    print("绝对值误差：", np.abs(y_true - diff_value))
    red.plt_differentiation([1, 5], dfun, x0, diff_value, is_show=False, is_fh_marker=True)
plt.show()
print("=" * 80)

# (2) 多点公式法二阶数值微分
d2fun = sympy.lambdify(x, sympy.diff(fun_sym, x, 2))  # 二阶导函数
fpsod = FivePointsSecondOrderDifferentiation(diff_fun=fun, h=0.01)
diff_val = fpsod.fit_2d_diff(x0)
print("二阶近似微分：", diff_val)
y_true = d2fun(x0)  # 二阶导数真值
print("精确值：", y_true, "\n绝对值误差：", np.abs(y_true - diff_val))
print("=" * 80)

# (2) 三次均匀B样条函数法二阶数值微分
cbsod = CubicBSpline2OrderDifferentiation(fun, n=20, h=0.01)
diff_val = cbsod.predict_diff_x0(x0)
print("二阶近似微分：", diff_val)
print("绝对值误差：", np.abs(y_true - diff_val))
print("=" * 80)
cbsod.plt_2_order_different([1, 5], d2fun, is_show=True, is_fh_marker=True)
plt.show()