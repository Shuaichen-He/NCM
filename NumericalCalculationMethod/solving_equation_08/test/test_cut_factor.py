# -*- coding: UTF-8 -*-
"""
@file_name: test_cut_factor.py
@time: 2022-11-11
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from solving_equation_08.cut_factor_method import CutFactorMethod

P = np.array([1, -9, 19, 5, 12, -44, -80])
for pi in np.arange(-10, 10, 0.2):
    for qi in np.arange(-10, 10, 0.2):
        cfm = CutFactorMethod(P, p0=pi, q0=qi, eps=1e-16, max_iter=500)
        cfm.fit_cut_factor()
        if np.max(np.abs(np.imag(cfm.root))) < 1:
            print("初始值：", pi, qi)
            print("二次因子系数：", cfm.omega_x)
            print("根：", cfm.root[0], cfm.root[1])
            print("精度：", cfm.precision[0], cfm.precision[1])
            print("=" * 50)
# print(pq_vals)
