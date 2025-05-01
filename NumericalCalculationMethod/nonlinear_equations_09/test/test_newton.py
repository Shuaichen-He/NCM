# -*- coding: UTF-8 -*-
"""
@file:test_newton.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""

import numpy as np
import sympy
from nonlinear_equations_09.nlinequs_newton import NLinearFxNewton

x, y = sympy.symbols("x, y")
nlin_equs = [x + 2 * y - 3, 2 * x ** 2 + y ** 2 - 5]
# x0 = np.array([-2, 2])
x0 = np.array([1.5, 1])
mni = NLinearFxNewton(nlin_equs, [x, y], x0, max_iter=10, eps=1e-15, method="downhill", is_plt=True)
mni.fit_roots()
print(mni.iter_roots_precision[-1])
print(mni.fxs_precision)
print("下山因子：", mni.downhill_lambda)
print("=" * 80)

# x, y, z = sympy.symbols("x, y, z")
# nlin_equs = [3 * x - sympy.cos(y * z) - 0.5,
#              x ** 2 - 81 * (y + 0.1) ** 2 + sympy.sin(z) + 1.06,
#              sympy.exp(-x * y) + 20 * z + 10 / 3 * sympy.pi - 1]
# x0 = np.array([0.2, 0, -0.2])
# mni = NLinearFxNewton(nlin_equs, [x, y, z], x0, max_iter=100, eps=1e-15, method="downhill", is_plt=True)
# mni.fit_roots()
# for i in range(3):
#     print("%.20f, %.15e" % (mni.roots[i], mni.fxs_precision[i]))
# print("下山因子：", mni.downhill_lambda)

# x, y = sympy.symbols("x, y")
# nlin_equs = [0.5 * sympy.sin(x) + 0.1 * sympy.cos(x * y) - x,
#              0.5 * sympy.cos(x) - 0.1 * sympy.cos(y) - y]
# # nlin_equs = [x ** 2 - 10 * x + y ** 2 + 8, x * y ** 2 + x - 10 * y + 8]
# x0 = np.array([0, 0])
# mni = NLinearFxNewton(nlin_equs, [x, y], x0, max_iter=100, eps=1e-15, method="downhill", is_plt=True)
# mni.fit_nlinequs_roots()
