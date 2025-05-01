# -*- coding: UTF-8 -*-
"""
@file_name: test_neuman_boundary_problem.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""

import numpy as np
import matplotlib.pyplot as plt
from ode_numerical_solution_11.ode_neumann_boundary_problem import ODENeumannBoundaryProblem

# （1）ODE边值问题
ode_f = lambda t: np.exp(t) * (np.sin(t) - 2 * np.cos(t))
q_x = lambda t: 1 + 0.0 * t
lambda_1, lambda_2 = 0, 0
u_t0, u_tm, t_T = -1, -np.exp(np.pi), np.pi
ode_ux = lambda t: np.exp(t) * np.sin(t)  # 解析解

# （2）ODE边值问题
# ode_f = lambda t: np.exp(t) * np.sin(t)
# q_x = lambda t: 1 + np.sin(t)
# lambda_1, lambda_2 = 1, 2
# u_t0, u_tm, t_T = 0, 3 * np.exp(1), 1
# ode_ux = lambda t: np.exp(t)  # 解析解

plt.figure(figsize=(14, 5))
h_array = np.array([np.pi / 100, np.pi / 200, np.pi / 300, np.pi / 400])
# h_array = np.array([1 / 20, 1 / 40, 1 / 80, 1 / 160])
h_labels = ["\pi/100", "\pi/200", "\pi/300", "\pi/400"]
# h_labels = ["1/100", "1/200", "1/300", "1/400"]
line_style = ["r:", "g--", "c-.", "k-"]
ode_bp = None
plt.subplot(121)
for i, (h, line) in enumerate(zip(h_array, line_style)):
    ode_bp = ODENeumannBoundaryProblem(ode_f, q_x, lambda_1, lambda_2, u_t0, u_tm, t_T, h)
    ode_bp.fit_ode()
    t_i = np.arange(0, t_T + h, h)
    sol_ux = ode_ux(t_i)
    error = sol_ux - ode_bp.ode_sol[:, 1]
    eps = np.linalg.norm(error)
    print(eps)
    plt.semilogy(t_i, np.abs(error), line, lw=1.5, label="$h = %s, \ \epsilon = %.2e$" % (h_labels[i], eps))
plt.xlabel("$x$", fontdict={"fontsize": 18})
plt.ylabel(r"$err = \vert y_k - \hat y_k \vert$", fontdict={"fontsize": 18})
plt.title("一阶$ODE \ Neumann$边值问题误差曲线 $\epsilon = \Vert y - \hat{y} \Vert_2$", fontdict={"fontsize": 18})
plt.legend(frameon=False, fontsize=16)
plt.grid(ls=":")
plt.tick_params(labelsize=18)  # 刻度字体大小16
plt.subplot(122)
ode_bp.plt_ode_numerical_sol(ode_analytical=ode_ux, is_show=False)  # 绘制最后一次步长的微分值图象
plt.show()
