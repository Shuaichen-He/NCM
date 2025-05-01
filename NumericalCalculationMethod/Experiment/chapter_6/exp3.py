# -*- coding: UTF-8 -*-
"""
@file_name: exp3.py
@time: 2023-02-10
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from direct_solution_linear_equations_06.chasing_method_tridiagonal_matrix \
    import ChasingMethodTridiagonalMatrix

d = np.array([3, -3, -10, 2])  # 右端向量
a = np.array([1, 3, 2])  # 次对角线元素，对角线以下
b = np.array([2, 2, -7, 5])  # 主对角线元素
c = np.array([1, -3, 4])  # 次对角线元素，对角线以上

sol_method = ["gauss", "doolittle", "crout"]
for method in sol_method:
    cmtm = ChasingMethodTridiagonalMatrix(diag_a=a, diag_b=b, diag_c=c, d_vector=d, sol_method=method)
    cmtm.fit_solve()
    print("y：", cmtm.y)
    print("x：", cmtm.x)
    print("eps：", cmtm.eps)
    print("=" * 60)