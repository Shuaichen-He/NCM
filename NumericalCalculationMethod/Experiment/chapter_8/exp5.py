# -*- coding: UTF-8 -*-
"""
@file_name: exp5.py
@time: 2023-02-11
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from solving_equation_08.cut_factor_method import CutFactorMethod

P = np.array([16, -40, 5, 20, 6])
cfm = CutFactorMethod(P, p0=1, q0=3)
cfm.fit_cut_factor()
print("二次因子系数：", cfm.omega_x)
print("根：", cfm.root)
print("精度：", cfm.precision)
print("=" * 50)
