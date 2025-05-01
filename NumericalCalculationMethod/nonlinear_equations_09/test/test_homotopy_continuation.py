# -*- coding: UTF-8 -*-
"""
@file_name: test_homotopy_continuation.py
@time: 2022-12-29
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import sympy
from nonlinear_equations_09.homotopy_continuation_method import HomotopyContinuationMethod

x, y, z = sympy.symbols("x, y, z")
nlin_equs = [3 * x - sympy.cos(y * z) - 0.5,
             x ** 2 - 81 * (y + 0.1) ** 2 + sympy.sin(z) + 1.06,
             sympy.exp(-x * y) + 20 * z + (10 * sympy.pi - 3) / 3]

nlin_funs2 = [sympy.sin(x) + y ** 2 + sympy.log(z) - 7,
              3 * x + 2 ** y - z ** 3 + 1,
              x + y + z - 5]

x0 = np.array([0, 0, 0])
hcm = HomotopyContinuationMethod(nlin_equs, [x, y, z], x0, N=4, method="continuation")
x, eps = hcm.fit_roots()
for i in range(3):
    print("%.16f, %.10e" % (x[i], eps[i]))


# x0 = np.array([0, 2, 1])
# hcm2 = HomotopyContinuationMethod(nlin_funs2, [x, y, z], x0, N=4)
# x, eps = hcm2.fit_roots()
# for i in range(3):
#     print("%.16f, %.10e" % (x[i], eps[i]))
