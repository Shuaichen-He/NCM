# -*- coding: UTF-8 -*-
"""
@file_name: test_pde_2d_p.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
from iterative_solution_linear_equation_07.pde_solve_2d_poisson_test import PDESolvePoisson2dModel


if __name__ == '__main__':
    poisson = PDESolvePoisson2dModel([0, 1], [0, 1], 10, 10, is_show=True, is_exact_fun=True)
    poisson.fit_pde()