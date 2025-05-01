# -*- coding: UTF-8 -*-
"""
@file_name: exp3.py
@time: 2023-02-11
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import matplotlib.pyplot as plt
from iterative_solution_linear_equation_07.conjugate_gradient_method import ConjugateGradientMethod
from iterative_solution_linear_equation_07.steepest_descent_method import SteepestDescentMethod
from iterative_solution_linear_equation_07.pre_conjugate_gradient import PreConjugateGradient

A = -1 * np.array([[-3, 1, 0, 0, 0, 0.5], [1, -3, 1, 0, 0, 0], [0, 1, -3, 1, 0, 0],
                   [0, 0, 1, -3, 1, 0], [0, 0, 0, 1, -3, 1], [0.5, 0, 0, 0, 1, -3]])
b = np.array([2.5, 1.5, 1, 1, 1.5, 2.5])
x0 = np.zeros(6)

sdm = SteepestDescentMethod(A, b, x0, eps=1e-14, is_out_info=True)  # 最速下降法
sdm.fit_solve()
print("=" * 80)

cgm = ConjugateGradientMethod(A, b, x0, eps=1e-14, is_out_info=True)  # 共轭梯度法
cgm.fit_solve()
print("=" * 80)

pcg = PreConjugateGradient(A, b, x0, eps=1e-14, is_out_info=True)   # 预处理共轭梯度法
pcg.fit_solve()

# 可视化
plt.figure(figsize=(14, 5))
plt.subplot(121)
sdm.plt_convergence_x(is_show=False, style="go-")
plt.subplot(122)
cgm.plt_convergence_x(is_show=False)
plt.show()

