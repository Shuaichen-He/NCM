# -*- coding: UTF-8 -*-
"""
@file:test_2order_diff_multipotins.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import sympy
from numerical_differentiation_05.five_points_second_order_differentiation \
    import FivePointsSecondOrderDifferentiation

fun = lambda x: np.sin(x) / np.sqrt(x)  # 微分函数


def second_derivative_fun():
    x = sympy.Symbol("x")
    fun = sympy.sin(x) / sympy.sqrt(x)  # 符号定义
    diff_fun = fun.diff(x, 2)  # 2阶导数
    fun_expr = sympy.lambdify(x, diff_fun, "numpy")  # lambda函数，进行数值运算
    return fun_expr


x0 = np.linspace(2, 5, 11)  # 待求微分的数据点
fpsod = FivePointsSecondOrderDifferentiation(diff_fun=fun, h=0.01)
diff_val = fpsod.fit_2d_diff(x0)
print("二阶微分：", diff_val)
fun_2d = second_derivative_fun()
y_true = fun_2d(x0)
print("精确值：", y_true, "\n误差：", y_true - diff_val)
for i in range(len(x0)):
    print("二阶微分：%.10f，精确值：%.10f，误差：%.10e" % (diff_val[i], y_true[i], y_true[i] - diff_val[i]))
print("-" * 80)
# 2. 离散数据形式计算二阶导数
fpsod = FivePointsSecondOrderDifferentiation(x=x0, y=fun(x0))
diff_val = fpsod.fit_2d_diff(x0)
print("二阶微分：", diff_val, "\n误差：", y_true - diff_val)
for i in range(len(x0)):
    print("二阶微分：%.10f，精确值：%.10f，误差：%.10e" % (diff_val[i], y_true[i], y_true[i] - diff_val[i]))
# y_true = fun_2d(xi)
# print("准确值：",)
# print("误差：", y_true - diff_val)
