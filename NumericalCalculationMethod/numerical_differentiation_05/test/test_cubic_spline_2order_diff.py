# -*- coding: UTF-8 -*-
"""
@file_name: test_cubic_spline_2order_diff.py
@time: 2021-11-25
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import sympy
from numerical_differentiation_05.cubic_spline_2_order_differentiation \
    import CubicSpline2OrderDifferentiation
from numerical_differentiation_05.cubic_hermite_2_order_differentiation \
    import CubicHermite2OrderDifferentiation

fun = lambda x: np.sin(x) / np.sqrt(x)  # 微分函数


def second_derivative_fun():
    x = sympy.Symbol("x")
    fun = sympy.sin(x) / sympy.sqrt(x)  # 符号定义
    diff_fun = fun.diff(x, 2)  # 2阶导数
    fun_expr = sympy.lambdify(x, diff_fun, "numpy")  # lambda函数，进行数值运算
    return fun_expr


x = np.linspace(2, 5, 50)
x0 = np.array([2, 2.3, 2.6, 2.9, 3.2, 3.5, 3.8, 4.1, 4.4, 4.7, 5 ])
print(second_derivative_fun()(x0))
cssod = CubicSpline2OrderDifferentiation(x, fun(x))
diff_value = cssod.cal_diff(x0)
print("微分值：", diff_value, "\n误差：", diff_value - second_derivative_fun()(x0))
print("=" * 80)
cssod = CubicHermite2OrderDifferentiation(x, fun(x))
diff_value = cssod.cal_diff(x0)
print("微分值：", diff_value, "\n误差：", diff_value - second_derivative_fun()(x0))