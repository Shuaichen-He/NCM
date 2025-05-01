# -*- coding: UTF-8 -*-
"""
@file_name: test_sd_cg.py
@time: 2022-11-09
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import matplotlib.pyplot as plt
from iterative_solution_linear_equation_07.steepest_descent_method import SteepestDescentMethod
from iterative_solution_linear_equation_07.conjugate_gradient_method import ConjugateGradientMethod
from iterative_solution_linear_equation_07.pre_conjugate_gradient import PreConjugateGradient

np.random.seed(0)
n = 1000
B = np.random.rand(n, n)
A = B.T * B + 20 * np.eye(n)
b = np.random.rand(n)
x0 = np.zeros(n)

sdm = PreConjugateGradient(A, b, x0, eps=1e-12, max_iter=1000, is_out_info=False)
sdm.fit_solve()
plt.figure(figsize=(14, 5))
plt.subplot(121)
sdm.plt_convergence_x(is_show=False, style="go-")

cgm = ConjugateGradientMethod(A, b, x0, eps=1e-12, is_out_info=True)
cgm.fit_solve()
plt.subplot(122)
cgm.plt_convergence_x(is_show=False)
plt.show()
