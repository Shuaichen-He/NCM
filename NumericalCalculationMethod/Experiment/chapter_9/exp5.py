# -*- coding: UTF-8 -*-
"""
@file_name: exp5.py
@time: 2023-02-11
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import sympy
import numpy as np
from nonlinear_equations_09.homotopy_continuation_method import HomotopyContinuationMethod

x_init = np.array([[0, 0, 0], [1, 0, -1]])
N_num = [4, 8, 10]
for x0 in x_init:
    for n in N_num:
        x, y, z = sympy.symbols("x, y, z")
        nlin_funs = [3 * x - sympy.cos(y * z) - 0.5,
                     4 * x ** 2 - 625 * y ** 2 + 2 * y - 1,
                     sympy.exp(-x * y) + 20 * z + 10 * sympy.pi / 3 - 1]
        hcm = HomotopyContinuationMethod(nlin_funs, [x, y, z], x0, N=n, method="newton")
        x, eps = hcm.fit_roots()
        print(x0, n)
        for i in range(3):
            print("x = %.16f, %.10e" % (x[i], eps[i]))
        print("=" * 60)
