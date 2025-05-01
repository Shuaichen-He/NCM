# -*- coding: UTF-8 -*-
"""
@file_name: ode_neumann_boundary_problem.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from direct_solution_linear_equations_06.chasing_method_tridiagonal_matrix \
    import ChasingMethodTridiagonalMatrix
from util_font import *


class ODENeumannBoundaryProblem:
    """
    ODE边值问题求解，导数边界值条件
    """

    def __init__(self, ode_fun, q_t, lambda_1, lambda_2, u_t0, u_tm, t_T, h=0.1):
        self.ode_fun = ode_fun  # 待求解的微分方程的右端项
        self.q_t = q_t  # u(x)的系数或函数，函数定义形式
        # 导数边界条件的系数或函数，函数定义形式
        self.lambda_1, self.lambda_2 = lambda_1, lambda_2
        self.u_t0, self.u_tm = u_t0, u_tm  # 边界值
        self.t_T = t_T  # 求解区间的终点
        self.h = h  # 求解步长
        self.ode_sol = None  # 求解的微分数值解

    def fit_ode(self):
        """
        求解ODE边界值问题，导数边界边界
        :return:
        """
        t_array = np.arange(0, self.t_T + self.h, self.h)  # 待求解ode区间的离散数值
        n = len(t_array)  # 离散点的个数
        self.ode_sol = np.zeros((n, 2))  # ode的数值解
        self.ode_sol[:, 0] = t_array
        self.ode_sol[0, 1], self.ode_sol[-1, 1] = self.u_t0, self.u_tm  # 边界值
        f_t = self.h ** 2 * self.ode_fun(t_array)  # 右端项的值， n+1
        # 第一项和最后一项特殊处理
        f_t[0]= f_t[0] / 2 + self.h * self.u_t0
        f_t[-1] = f_t[-1] / 2 + self.h * self.u_tm
        # 如下构造系数矩阵
        miu_i = self.h ** 2 * self.q_t(t_array)
        a_diag = np.zeros(n)
        a_diag[0] = 1 + self.h * self.lambda_1 + miu_i[0] / 2
        a_diag[-1] = 1 + self.h * self.lambda_2 + miu_i[-1] / 2
        a_diag[1:-1] = 2 + miu_i[1:-1]  # 主对角线元素
        b_diag = -1 * np.ones(n - 1)
        # 用追赶法求解
        cmtm = ChasingMethodTridiagonalMatrix(b_diag, a_diag, b_diag, f_t)
        self.ode_sol[:, 1] = cmtm.fit_solve()
        return self.ode_sol

    def plt_ode_numerical_sol(self, is_show=True, ode_analytical=None):
        """
        可视化ode数值解曲线
        :return:
        """
        if is_show:
            plt.figure(figsize=(8, 6))
        plt.plot(self.ode_sol[:, 0], self.ode_sol[:, 1], "k-", lw=1.5,
                 label="$Numerical \ Sol \ \hat y(x) \ (h = %.2e)$" % self.h)
        if ode_analytical:
            ode_res = ode_analytical(self.ode_sol[:, 0])
            eps = np.linalg.norm(self.ode_sol[:, 1] - ode_res)
            plt.plot(self.ode_sol[:, 0], ode_res, "r--", lw=1.5,
                     label="$Analytical \ Sol \ y(x) \ (\epsilon = %.2e)$" % eps)
        plt.xlabel("$x$", fontdict={"fontsize": 18})
        plt.ylabel("$y(x) \ / \ \hat y(x)$", fontdict={"fontsize": 18})
        plt.title("一阶$ODE \ Neumann$边值问题数值解与解析解曲线", fontdict={"fontsize": 18})
        plt.legend(frameon=False, fontsize=16)
        plt.grid(ls=":")
        plt.tick_params(labelsize=18)  # 刻度字体大小16
        if is_show:
            plt.show()
