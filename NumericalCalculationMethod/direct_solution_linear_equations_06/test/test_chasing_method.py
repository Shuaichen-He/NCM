# -*- coding: UTF-8 -*-
"""
@file_name: test_chasing_method.py
@time: 2021-09-29
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from direct_solution_linear_equations_06.chasing_method_tridiagonal_matrix \
    import ChasingMethodTridiagonalMatrix

# d = np.array([10, 15, 20, 20, 10])
# a = np.array([-1, -1, -1, -1])
# b = np.array([4, 4, 4, 4, 4])
# c = np.array([-1, -1, -1, -1])
# d = np.array([7, 11, 15, 9])
# a = np.array([-1, -1, -1])
# b = np.array([3, 3, 3, 3])
# c = np.array([2, 2, 2])
n = 50  # 维度
b = 4 * np.ones(n)
a = -1 * np.ones(n - 1)
c = -1 * np.ones(n - 1)
d = 10 * np.ones(n)  # 问题(1)的右端向量
sol_method = ["gauss", "doolittle", "crout"]
for method in sol_method:
    cmtm = ChasingMethodTridiagonalMatrix(a, b, c, d_vector=d, sol_method=method)
    cmtm.fit_solve()
    print("y：", cmtm.y)
    print("x：", cmtm.x)
    print("eps：", cmtm.eps)
    print("=" * 60)
