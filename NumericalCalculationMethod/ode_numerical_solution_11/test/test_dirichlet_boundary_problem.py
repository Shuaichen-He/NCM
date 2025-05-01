# -*- coding: UTF-8 -*-
"""
@file_name: test_dirichlet_boundary_problem.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""

import numpy as np
from ode_numerical_solution_11.ode_dirichlet_boundary_problem import ODEDirichletBoundaryProblem
from util_font import *


# （1）ODE边值问题
ode_f = lambda t: np.exp(t) * (np.sin(t) - 2 * np.cos(t))
q_x = lambda t: 1 + 0.0 * t
u_t0, u_tm, t_T = 0, 0, np.pi
ode_ux = lambda t: np.exp(t) * np.sin(t)  # 解析解

# （2）ODE边值问题
# ode_f = lambda t: (t ** 2 - t + 1.25) * np.sin(t)
# q_x = lambda t: (t - 0.5) ** 2
# u_t0, u_tm, t_T = 0, 1, np.pi / 2
# ode_ux = lambda t: np.sin(t)  # 解析解

# ode_f = lambda t: (1 / 20 * t ** 4 - 6) * t
# q_x = lambda t: 1.0 + t * 0.0
# u_t0, u_tm, t_T = 0, 21 / 20, 1
# ode_ux = lambda t: 1 / 20 * t ** 5 + t ** 3  # 解析解

plt.figure(figsize=(14, 5))
h_array = np.array([np.pi / 20, np.pi / 40, np.pi / 80, np.pi / 160])
h_labels = ["\pi/20", "\pi/40", "\pi/80", "\pi/160"]
plt.subplot(121)
line_style = ["ro:", "g*--", "c+-.", "k-"]
ode_bp = None
for i, (h, line) in enumerate(zip(h_array, line_style)):
    ode_bp = ODEDirichletBoundaryProblem(ode_f, q_x, u_t0, u_tm, t_T, h, diff_type="basic")  # basic, compact
    ode_bp.fit_ode()
    t_i = np.arange(0, t_T + h, h)
    sol_ux = ode_ux(t_i)
    error = ode_bp.ode_sol[:, 1] - sol_ux
    eps = np.linalg.norm(error)
    print(h, eps)
    plt.semilogy(t_i[1:-1], np.abs(error[1:-1]), line, lw=1.5, label="$h = %s, \ \epsilon = %.2e$" % (h_labels[i], eps))
plt.xlabel("$x$", fontdict={"fontsize": 18})
plt.ylabel(r"$err = \vert y_k - \hat y_k \vert$", fontdict={"fontsize": 18})
plt.title("一阶$ODE \ Dirichlet$边值问题误差曲线 $\epsilon = \Vert y - \hat{y} \Vert_2$", fontdict={"fontsize": 18})
plt.legend(frameon=False, fontsize=16, loc="lower center")
plt.grid(ls=":")
plt.tick_params(labelsize=18)  # 刻度字体大小16
plt.subplot(122)
ode_bp.plt_ode_numerical_sol(ode_analytical=ode_ux, is_show=False)  # 绘制最后一次步长的数值解图象
plt.show()