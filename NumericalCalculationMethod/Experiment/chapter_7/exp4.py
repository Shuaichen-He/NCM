# -*- coding: UTF-8 -*-
"""
@file_name: exp4.py
@time: 2023-02-11
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
from Experiment.chapter_7.pde_solve_2d_poisson_test import PDESolvePoisson2dModel

poisson = PDESolvePoisson2dModel([0, 1], [0, 1], 10, 10, is_show=True, is_exact_fun=True)
poisson.fit_pde()

poisson = PDESolvePoisson2dModel([0, 1], [0, 1], 20, 20, is_show=True, is_exact_fun=True)
poisson.fit_pde()
