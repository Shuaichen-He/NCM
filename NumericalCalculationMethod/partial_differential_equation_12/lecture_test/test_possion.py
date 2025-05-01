# -*- coding: UTF-8 -*-
"""
@file_name: test_possion.py
@time: 2023-05-08
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from util_font import *
from partial_differential_equation_12.lecture_test.pde_poisson_equation_matrix import PDEPoissonEquationTriBMatrix


g_xy = lambda x, y: 2 * np.pi ** 2 * np.sin(np.pi * x) * np.sin(np.pi * y)  # 右端函数
f_u0y, f_uay = lambda y: 0, lambda y: 0
f_ux0, f_uxb = lambda x: 0, lambda x: 0
x_span, y_span, n_x, n_y = [0, 1], [0, 1], 16, 16
pde_model = lambda x, y: np.sin(np.pi * x) * np.sin(np.pi * y)


poisson = PDEPoissonEquationTriBMatrix(g_xy, f_ux0, f_uxb, f_u0y, f_uay, x_span, y_span, n_x, n_y,
                                       pde_model=pde_model, is_show=True)
poisson.solve_pde()
poisson.plt_pde_poisson_surface()
poisson.plt_convergence_precision(is_show=True)
plt.figure(figsize=(7, 5))
ax = plt.gca(projection='3d')
xi = np.linspace(0, 1, 100)
yi = np.linspace(0, 1, 100)
y, x = np.meshgrid(yi, xi)
z = pde_model(x, y)
ax.plot_surface(x, y, z, cmap='rainbow')
ax.set_xlabel("$x$", fontdict={"fontsize": 18})
ax.set_ylabel("$y$", fontdict={"fontsize": 18})
ax.set_zlabel("$U$", fontdict={"fontsize": 18})
plt.tick_params(labelsize=16)  # 刻度字体大小16
plt.title("$Poisson$方程解析解曲面", fontdict={"fontsize": 18})
plt.show()