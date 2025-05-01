# -*- coding: UTF-8 -*-
"""
@file_name: test_comp_2d_general_region_12.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import sympy
from numerical_integration_04.com_simpson_2d_general_region import GeneralRegionDoubleIntegration

x, y = sympy.symbols("x, y")
int_fun = 3 * x ** 2 * y ** 2
c_x, d_x = 0, 1 - x ** 2
grdi = GeneralRegionDoubleIntegration(int_fun, np.array([0, 1]), c_x, d_x, interval_num=200)  # 修改区间划分数
int_value = grdi.fit_2d_int()
print(int_value)
print("误差：", 16 / 315 - int_value)
