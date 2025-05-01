# -*- coding: UTF-8 -*-
"""
@file:test_gauss_chebyshev_6.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from numerical_integration_04.gauss_chebyshev_int import GaussChebyshevIntegration

fun = lambda x: np.exp(x)  # 第一类切比雪夫示例被积函数
gci = GaussChebyshevIntegration(fun, 10, cb_type=1)  # 修改零点数 n + 1
gci.fit_int()
print("第一类切比雪夫多项式零点：", gci.zero_points)
print("插值型求积系数：", gci.A_k)
print("积分值：", gci.int_value, "误差：", 3.9774632605064228 - gci.int_value)

print("=" * 80)

fun = lambda x: x ** 2  # 第二类切比雪夫示例被积函数
gci = GaussChebyshevIntegration(fun, 16, cb_type=2)  # 修改零点数 n
gci.fit_int()
print("第二类切比雪夫多项式零点：", gci.zero_points)
print("插值型求积系数：", gci.A_k)
print("积分值：", gci.int_value, "误差：", np.pi / 8 - gci.int_value)
