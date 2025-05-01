# -*- coding: UTF-8 -*-
"""
@file_name: test_mid_point_diff.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import matplotlib.pyplot as plt
import sympy
from numerical_differentiation_05.middle_point_formula_differentiation \
    import MiddlePointFormulaDifferentiation


x = sympy.Symbol("x")
fun = sympy.exp(-0.5 * x) * sympy.sin(2 * x)  # 被微分函数
x0 = np.array([1.23, 1.75, 1.89, 2.14, 2.56])  # 求解指定点的微分
mpfd = MiddlePointFormulaDifferentiation(fun, h=0.001, is_error=True)  # 实例化
y0 = mpfd.fit_diff(x0)  # 中点公式求解微分
diff_val = sympy.lambdify(x, fun.diff(x, 1))(x0)  # 函数一阶导在x0的函数值，用于误差分析
print("微分值：", y0, "\n精确值：", diff_val, "\n误差：", diff_val - y0)
print("截断误差：", mpfd.diff_error)

h_vector = [0.2, 0.001]  # 不同微分步长
plt.figure(figsize=(14, 5))
for i, h in enumerate(h_vector):
    plt.subplot(121 + i)
    mpfd = MiddlePointFormulaDifferentiation(fun, h=h, is_error=False)
    y0 = mpfd.fit_diff(x0)
    if h == 0.001:
        mpfd.plt_differentiation([1, 3], x0, y0, is_show=False, is_fh_marker=True)
    else:
        mpfd.plt_differentiation([1, 3], x0, y0, is_show=False)
plt.show()
