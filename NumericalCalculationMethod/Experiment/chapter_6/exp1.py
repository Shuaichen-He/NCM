# -*- coding: UTF-8 -*-
"""
@file_name: exp1.py
@time: 2023-02-10
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from direct_solution_linear_equations_06.gaussian_elimination_algorithm import GaussianEliminationAlgorithm

A = np.array([[12, -3, 3, 4], [-18, 3, -1, -1], [1, 1, 1, 1], [3, 1, -1, 1]])
b = np.array([15, -15, 6, 2])
sol_method = ["sequential", "column", "complete", "jordan"]
for method in sol_method:
    gea = GaussianEliminationAlgorithm(A, b, sol_method=method)
    gea.fit_solve()
    print(method + "，消元后的矩阵：\n", gea.augmented_matrix)
    print(method + "，线性方程组的解：\n", gea.x)
    print(method + "，线性方程组的解验证误差：\n", gea.eps)
    if method == "jordan":
        print("Jordan消元法，系数矩阵的逆矩阵是：\n", gea.jordan_inverse_matrix)
        print("Jordan消元法，逆矩阵验证：\n", np.dot(A, gea.jordan_inverse_matrix))
    print("=" * 80)
