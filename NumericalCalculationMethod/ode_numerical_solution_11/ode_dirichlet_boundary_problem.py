# -*- coding: UTF-8 -*-
"""
@file_name: ode_dirichlet_boundary_problem.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from direct_solution_linear_equations_06.chasing_method_tridiagonal_matrix \
    import ChasingMethodTridiagonalMatrix
from util_font import *


class ODEDirichletBoundaryProblem:
    """
    ODE边界问题求解，Dirichlet边界
    """
    def __init__(self, ode_fun, q_t, u_t0, u_tm, t_T, h=0.1, diff_type="compact"):
        self.ode_fun = ode_fun  # 待求解的微分方程的右端项
        self.q_t = q_t  # u(x)的系数或函数，函数定义形式
        self.u_t0, self.u_tm = u_t0, u_tm  # 边界值
        self.t_T = t_T  # 求解区间的终点
        self.h = h  # 求解步长
        self.diff_type = diff_type  # 古典格式和紧差分格式
        self.ode_sol = None  # 求解的微分数值解

    def fit_ode(self):
        """
        求解ODE边界值问题，Dirichlet边界
        :return:
        """
        t_array = np.arange(0, self.t_T + self.h, self.h)  # 待求解ode区间的离散数值
        n = len(t_array)  # 离散点的个数
        self.ode_sol = np.zeros((n, 2))  # ode的数值解
        self.ode_sol[:, 0] = t_array  # 第一列存储离散待递推数值
        self.ode_sol[0, 1], self.ode_sol[-1, 1] = self.u_t0, self.u_tm  # 边界值
        if self.diff_type.lower() == "compact":  # 紧差分格式
            self._solve_ode_compact_(n, t_array)
        elif self.diff_type.lower() == "basic":  # 基本差分格式
            self._solve_ode_basic_form_(n, t_array)
        else:
            raise ValueError("仅支持基本差分格式basic和紧差分格式compact.")

    def _solve_ode_basic_form_(self, n, t_array):
        """
        基本形式求解
        :return:
        """
        f_t = self.h ** 2 * self.ode_fun(t_array[1:-1])  # 右端项的值， n-1
        # 第一项和最后一项特殊处理
        f_t[0], f_t[-1] = f_t[0] + self.u_t0, f_t[-1] + self.u_tm
        a_diag = 2 + self.h ** 2 * self.q_t(t_array[1:-1])  # 主对角线元素
        b_diag = -1 * np.ones(n - 3)
        # 用追赶法求解
        cmtm = ChasingMethodTridiagonalMatrix(b_diag, a_diag, b_diag, f_t)
        self.ode_sol[1:-1, 1] = cmtm.fit_solve()
        return self.ode_sol

    def _solve_ode_compact_(self, n, t_array):
        """
        紧差分格式
        :return:
        """
        # 右端项的值和q(x)的值， n + 1
        f_t, qt_val = self.ode_fun(t_array), self.q_t(t_array)
        # 右端向量，n - 1
        b_vector = self.h ** 2 * (f_t[:-2] + 10 * f_t[1:-1] + f_t[2:]) / 12
        b_vector[0] -= self.u_t0 * (self.h ** 2 / 12 * qt_val[0] - 1)
        b_vector[-1] -= self.u_tm * (self.h ** 2 / 12 * qt_val[-1] - 1)
        a_diag = 2 + 5 * self.h ** 2 / 6 * qt_val[1:-1]  # 主对角线元素
        b_diag = (self.h ** 2 / 12 * qt_val[1:-2] - 1) * np.ones(n - 3)  # 主对角线以下
        c_diag = (self.h ** 2 / 12 * qt_val[2:-1] - 1) * np.ones(n - 3)  # 主对角线以上
        # 用追赶法求解
        cmtm = ChasingMethodTridiagonalMatrix(b_diag, a_diag, c_diag, b_vector,
                                              sol_method="doolittle")
        self.ode_sol[1:-1, 1] = cmtm.fit_solve()
        return self.ode_sol

    def plt_ode_numerical_sol(self, is_show=True, ode_analytical=None):
        """
        可视化ode数值解曲线
        :return:
        """
        if is_show:
            plt.figure(figsize=(7, 5))
        plt.plot(self.ode_sol[:, 0], self.ode_sol[:, 1], "k-", lw=1.5,
                 label="$Numerical \ Sol \ \hat y(x) \ (h = \pi/160)$")
        if ode_analytical:
            ode_res = ode_analytical(self.ode_sol[:, 0])
            eps = np.linalg.norm(self.ode_sol[:, 1] - ode_res)
            plt.plot(self.ode_sol[:, 0], ode_res, "r--", lw=1.5,
                     label="$Analytical \ Sol \ y(x) \ (\epsilon = %.2e)$" % eps)
        plt.xlabel("$x$", fontdict={"fontsize": 18})
        plt.ylabel("$y(x) \ / \ \hat y(x)$", fontdict={"fontsize": 18})
        plt.title("一阶$ODE \ Dirichlet$边值问题数值解与解析解曲线", fontdict={"fontsize": 18})
        plt.legend(frameon=False, fontsize=16)
        plt.grid(ls=":")
        plt.tick_params(labelsize=18)  # 刻度字体大小16
        if is_show:
            plt.show()
