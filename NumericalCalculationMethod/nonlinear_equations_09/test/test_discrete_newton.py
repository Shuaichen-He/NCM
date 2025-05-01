# -*- coding: UTF-8 -*-
"""
@file:test_discrete_newton.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from nonlinear_equations_09.nlinequs_discrete_newton import NLinearFxDiscreteNewton


def nlin_fx(x):
    y = [3 * x[0] - np.cos(x[1] * x[2]) - 0.5,
         x[0] ** 2 - 81 * (x[1] + 0.1) ** 2 + np.sin(x[2]) + 1.06,
         np.exp(-x[0] * x[1]) + 20 * x[2] + 10 / 3 * np.pi - 1]
    return np.asarray(y, dtype=np.float64)


h = [0.01, 0.01, 0.001]
x0 = np.array([2, 0, -5])
mdn = NLinearFxDiscreteNewton(nlin_fx, x0, h=h, eps=1e-15, max_iter=200, is_plt=True)
mdn.fit_roots()
print(mdn.downhill_lambda)
rp = mdn.iter_roots_precision[-1]
for i in range(len(rp[1])):
    print("%.25f, %.15e" % (rp[1][i], mdn.fxs_precision[i]))
