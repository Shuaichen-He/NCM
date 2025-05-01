# -*- coding: UTF-8 -*-
"""
@file:test_comp_simpson_3d_int_13.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
from numerical_integration_04.composite_simpson_3d_integration import CompositeSimpsonTripleIntegration
import numpy as np

int_fun = lambda x, y, z: 4 * x * z * np.exp(-x ** 2 * y - z ** 2)

ci = CompositeSimpsonTripleIntegration(int_fun, [0, 1], [0, np.pi], [0, np.pi], 1e-10, max_split=200, increment=10)
ci.fit_3d_int()
print("积分近似值：%.15f, 误差：%15e" % (ci.int_value, 1.7327622230312205 - ci.int_value))
ci.plt_precision(exact_int=1.7327622230312205)

