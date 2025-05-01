# -*- coding: UTF-8 -*-
"""
@file_name: test_nelder_mead.py
@time: 2022-09-10
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import matplotlib.pyplot as plt
from numerical_optimization_13.nelder_mead_2d import NelderMeadOptimization

# fun = lambda x: x[0] ** 2 - 4 * x[0] + x[1] ** 2 - x[1] - x[0] * x[1]
# V_k = np.array([[0, 0], [1.2, 0.0], [0.0, 0.8]])
# fun = lambda x: (x[0] ** 2 - 2 * x[0]) * np.exp(-x[0] ** 2 - x[1] ** 2 - x[0] * x[1])  # 极小值
fun = lambda x: -1 * (x[0] ** 2 - 2 * x[0]) * np.exp(-x[0] ** 2 - x[1] ** 2 - x[0] * x[1])  # 极大值
V_k = np.array([[0, 0], [0.5, 1], [1.5, 1]])
eps = 1e-15
nmo = NelderMeadOptimization(fun, V_k, eps, is_minimum=False)  # 此处修改is_minimum为False求极大值
m_x = nmo.fit_optimize()
print("[(%.15f, %.15f), %.15f]" % (m_x[0], m_x[1], m_x[2]))
# nmo.plt_optimization([2, 4], [1, 3])
nmo.plt_optimization([-3, 3], [-3, 3])