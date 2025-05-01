# -*- coding: UTF-8 -*-
"""
@file_name: test_implicit_numerical_diff.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from numerical_differentiation_05.implicit_numerical_diff import ImplicitNumericalDifferentiation

fun = lambda x: np.sin(x) * np.exp(-x)
x = np.linspace(1, 5, 50)
np.random.seed(100)
x0 = np.sort(1 + 4 * np.random.rand(5))  # [1, 5]随机生成5个值
imp_nd = ImplicitNumericalDifferentiation(x, fun(x))
imp_nd.fit_diff()
dy0 = imp_nd.predict_x0(x0)  # 近似值
y_true = np.exp(-x0) * (np.cos(x0) - np.sin(x0))  # 精确微分值
print("所求微分值的点：", x0)
print("微分值：", dy0, "\n精确值：", y_true, "\n误差：", y_true - dy0)

