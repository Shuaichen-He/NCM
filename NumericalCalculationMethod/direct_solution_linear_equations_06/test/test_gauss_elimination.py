# -*- coding: UTF-8 -*-
"""
@file_name: test_gauss_elimination.py
@time: 2021-09-28
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""

import numpy as np
from direct_solution_linear_equations_06.gaussian_elimination_algorithm import GaussianEliminationAlgorithm


# A = np.array([[1, 1, 1], [0, 4, -1], [2, -2, 1]])
# b = np.array([6, 5, 1])
# A = np.array([[0.2641, 0.1735, 0.8642], [0.9411, -0.0175, 0.1463], [-0.8641, -0.4243, 0.0711]])
# b = np.array([-0.7521, 0.6310, 0.2501])
# A = np.array([[2, -3, 6], [-7, 10, 0], [-1, 5, 5]])
# b = np.array([4, 7, 6])
# A = np.array([[10, -19, -2], [-20, 40, 1], [1, 4, 5]])
# b = np.array([3, 4, 5])

A = np.array([[2, 5, 4, 1], [1, 3, 2, 1], [2, 10, 9, 7], [3, 8, 9, 2]])
b = np.array([20, 11, 40, 37])

# A = np.array([[12, -3, 3, 4], [-18, 3, -1, -1], [1, 1, 1, 1], [3, 1, -1, 1]])
# b = np.array([15, -15, 6, 2])
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
