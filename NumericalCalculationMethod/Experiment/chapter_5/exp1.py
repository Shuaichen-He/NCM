# -*- coding: UTF-8 -*-
"""
@file_name: exp1.py
@time: 2023-02-10
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import matplotlib.pyplot as plt
import sympy
from numerical_differentiation_05.middle_point_formula_differentiation \
    import MiddlePointFormulaDifferentiation
from numerical_differentiation_05.three_five_points_formula_differentiation \
    import ThreeFivePointsFormulaDifferentiation


def plt_differentiation(classname, fun, x0, points_type, diff_type):
    """
    可视化函数, param classname: 类名字符串
    """
    h_vector = [0.2, 0.001]  # 不同微分步长
    plt.figure(figsize=(14, 5))
    for i, h in enumerate(h_vector):
        plt.subplot(121 + i)
        if classname == "MiddlePointFormulaDifferentiation":
            obj = eval(classname)(fun, h=h, is_error=False)
        else:
            obj = eval(classname)(fun, h=h, points_type=points_type, diff_type=diff_type)
        y0 = mpfd.fit_diff(x0)
        if h == 0.001:
            obj.plt_differentiation([0, 5], x0, y0, is_show=False, is_fh_marker=True)
        else:
            obj.plt_differentiation([0, 5], x0, y0, is_show=False)
    plt.show()


# 中点公式法
x = sympy.Symbol("x")
fun = sympy.sin(x) / (1 + x) ** 2  # 被微分函数
x0 = np.array([0.21, 1.23, 1.75, 2.89, 3.14, 3.78, 4.11, 4.56])  # 求解指定点的微分
mpfd = MiddlePointFormulaDifferentiation(fun, h=0.01, is_error=False)  # 实例化
y0 = mpfd.fit_diff(x0)  # 中点公式求解微分
diff_val = sympy.lambdify(x, fun.diff(x, 1))(x0)  # 函数一阶导在x0的函数值，用于误差分析
print("微分值：", y0, "\n精确值：", diff_val, "\n绝对值误差：", abs(diff_val - y0))
print("=" * 80)
plt_differentiation("MiddlePointFormulaDifferentiation", fun, x0, None, None)

# 三点公式法
tfpfd = ThreeFivePointsFormulaDifferentiation(fun, h=0.01, points_type="three", diff_type="stirling")
y0 = tfpfd.predict_diff_x0(x0)
print("微分值：", y0, "\n精确值：", diff_val, "\n绝对值误差：", abs(diff_val - y0))
print("=" * 80)
plt_differentiation("ThreeFivePointsFormulaDifferentiation", fun, x0, "three", "stirling")

# 五点公式法
tfpfd = ThreeFivePointsFormulaDifferentiation(fun, h=0.01, points_type="five", diff_type="middle")
y0 = tfpfd.predict_diff_x0(x0)
print("微分值：", y0, "\n精确值：", diff_val, "\n绝对值误差：", abs(diff_val - y0))
print("=" * 80)
plt_differentiation("ThreeFivePointsFormulaDifferentiation", fun, x0, "five", "middle")
