# -*- coding: UTF-8 -*-
"""
@file:test_sor_newton.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from nonlinear_equations_09.nlinequs_newton_sor import NLinearFxNewtonSOR


def nlin_funs1(x):
    # 双曲型和圆
    nlinequs = [x[1] ** 2 - 4 * x[0] ** 2 + 1,
                x[0] ** 2 - 2 * x[0] + x[1] ** 2 - 3]
    return np.asarray(nlinequs, dtype=np.float64)

# x0 = np.array([[1, 2], [1, -2], [-1, 1], [-1, -1]])
# omega = [1, 1, 0.8, 0.8]
# h = [0.001, 0.001]
# for i in range(4):
#     msn = NLinearFxNewtonSOR(nlin_funs1, x0=x0[i], h=h, sor_factor=omega[i],
#                              max_iter=1000, eps=1e-15, method="sor", is_plt=True)
#     msn.fit_roots()
#     rp = msn.iter_roots_precision[-1]
#     print("迭代次数：", rp[0])
#     for i in range(len(rp[1])):
#         print("%.25f, %.15e" % (rp[1][i], msn.fxs_precision[i]))
#     print("=" * 60)

def nlin_funs2(x):
    # 立方体和抛物线
    nlinequs = [x[1] - x[0] ** 3 + 3 * x[0] ** 2 - 4 * x[0],
                x[1] ** 2 - x[0] - 2]
    return np.asarray(nlinequs, dtype=np.float64)


def nlin_funs3(x):
    n = 100
    nlinequs = np.zeros((n, 1))
    nlinequs[0] = (3 - 2 * x[0]) * x[0] - 2 * x[1] + 1
    nlinequs[-1] = (3 - 2 * x[-1]) * x[-1] - 2 * x[-2] + 1
    nlinequs[1:-1] = (2 - 2 * x[1:-1]) * x[1:-1] - x[:-2] - 2 * x[2:] + 1
    return np.asarray(nlinequs, dtype=np.float64)

x0 = -0.5 * np.ones(100)
h = 0.001 * np.ones(100)
msn = NLinearFxNewtonSOR(nlin_funs3, x0, h, jacobi_para=5, sor_factor=0.85, max_iter=1000,
                         eps=1e-15, method="sor", is_plt=True)
msn.fit_roots()
rp = msn.iter_roots_precision[-1]
for i in range(len(rp[1])):
    print("%.25f, %.15e" % (rp[1][i], msn.fxs_precision[i]))

idx = np.argmax(np.abs(msn.fxs_precision))
print("最大误差：", idx, msn.fxs_precision[idx])