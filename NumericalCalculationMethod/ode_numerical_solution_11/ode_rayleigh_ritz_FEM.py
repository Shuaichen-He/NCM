# -*- coding: UTF-8 -*-
"""
@file_name: ode_rayleigh_ritz_FEM.py
@time: 2023-02-17
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import sympy
from scipy import integrate
from numerical_integration_04.composite_quadrature_formula import \
    CompositeQuadratureFormula  # 复合科特斯求积公式
# 采用列主元高斯消元法求解线性方程组的解
from direct_solution_linear_equations_06.gaussian_elimination_algorithm \
    import GaussianEliminationAlgorithm  # 列主元高斯消元法
from util_font import *


class ODERayleighRitzFEM:
    """
    B样条基函数有限元法求解二阶ODE数值解, 带状稀疏矩阵. 采用复合科茨求积公式
    CompositeQuadratureFormul和列主元高斯消元法GaussianEliminationAlgorithm
    """

    def __init__(self, f_ux, p_x, q_x, x_span, n, ode_model=None):
        self.f_ux = f_ux  # 右端函数，符号函数定义，固定自变量符号为x
        self.p_x, self.q_x = p_x, q_x  # 方程中的函数项, 符号定义
        self.a, self.b = x_span[0], x_span[1]  # 求解区间，支持[a, b]形式
        self.n = n  # 子空间数
        self.h = (self.b - self.a) / (self.n + 1)  # 单元剖分步长
        self.ode_model = ode_model  # 解析解，不存在可不传
        self.ux = None  # 最终近似解表达式

    def basic_func(self, x, i):
        """
        B样条基函数的构造
        :param x: 符号变量
        :param i: 离散节点索引
        :return:
        """
        t = sympy.symbols("t")  # 定义符号变量
        t1, t2 = (2 + t) ** 3, (2 - t) ** 3  # 子项
        S_t = sympy.Piecewise((0.0, t <= -2),
                              (0.25 * t1, (t > -2) & (t <= -1)),
                              (0.25 * (t1 - 4 * (1 + t) ** 3), (t > -1) & (t <= 0)),
                              (0.25 * (t2 - 4 * (1 - t) ** 3), (t > 0) & (t <= 1)),
                              (0.25 * t2, (t > 1) & (t <= 2)),
                              (0.0, t > 2), (0.0, True))
        if i == 0:
            return S_t.subs({t: x / self.h}) - 4 * S_t.subs({t: (x / self.h + 1)})
        elif i == 1:
            return S_t.subs({t: (x / self.h - 1)}) - S_t.subs({t: (x / self.h + 1)})
        elif i == self.n:
            return S_t.subs({t: (x / self.h - self.n)}) - \
                   S_t.subs({t: (x / self.h - self.n - 2)})
        elif i == self.n + 1:
            return S_t.subs({t: (x / self.h - self.n - 1)}) - \
                   4 * S_t.subs({t: (x / self.h - self.n - 2)})
        else:
            return S_t.subs({t: (x / self.h - i)})

    def fit_ode(self):
        """
        核心算法：B样条基函数有限元法求解二阶ODE数值解
        :return:
        """

        xi = np.zeros(self.n + 6)  # 其中延拓四点x(-2)=x(-1)=0, x(n+2)=x(n+3)=1
        xi[[-2, -1]] = 1  # x(n+2)=x(n+3)=1
        xi[2:-2] = np.linspace(self.a, self.b, self.n + 2)  # 区间剖分点
        print(xi[2:-2])
        x = sympy.symbols("x")  # 定义符号变量
        # 如下构造求解近似解表达式的系数c的系数矩阵和右端向量
        A = np.zeros((self.n + 2, self.n + 2), dtype=np.float64)  # 带状系数矩阵
        b = np.zeros(self.n + 2, dtype=np.float64)  # 右端向量
        for i in range(self.n + 2):
            print(i, end=",")  # 打印计算的进度
            fai_x_i = self.basic_func(x, i)  # 基函数
            d_fai_x_i = fai_x_i.diff(x)  # 一阶导数
            for j in range(i, np.min([i + 3, self.n + 1]) + 1):
                L, U = np.max([xi[j], 0]), np.min([xi[i + 4], 1])  # 积分下限、上限
                fai_x_j = self.basic_func(x, j)  # 基函数
                d_fai_x_j = fai_x_j.diff(x)  # 一阶导数
                int_fun = self.p_x * d_fai_x_i * d_fai_x_j + \
                          self.q_x * fai_x_i * fai_x_j  # 被积函数符号形式
                # 复合科特斯积分算法
                cqf = CompositeQuadratureFormula(int_fun, [L, U], interval_num=40,
                                                 int_type="cotes")
                A[i, j] = cqf.fit_int()  # 计算积分
                if i != j:
                    A[j, i] = A[i, j]  # 对称矩阵
            if i >= 4:
                for j in range(i - 3):
                    A[i, j] = 0
            if i <= self.n - 3:
                for j in range(i + 4, self.n + 2):
                    A[i, j] = 0
            # 计算右端向量
            L, U = np.max([xi[i], 0]), np.min([xi[i + 4], 1])  # 积分下限、上限
            int_fun = self.f_ux * fai_x_i
            cqf = CompositeQuadratureFormula(int_fun, [L, U], interval_num=40,
                                             int_type="cotes")
            b[i] = cqf.fit_int()
        print("\n带状系数矩阵为：\n", A)
        print("右端向量：\n", b)
        gea = GaussianEliminationAlgorithm(A, b)  # 列主元高斯消元法
        gea.fit_solve()  # 可采用库函数求解c_sol = np.linalg.solve(A, b)
        c_sol = gea.x  # 线性方程组的解向量
        print("=" * 80)
        print("解向量：\n", c_sol)
        # 如下构造三次样条函数的和
        self.ux = sympy.zeros(1, self.n + 2)  # 用于存储基函数
        for i in range(self.n + 2):
            self.ux[i] = c_sol[i] * self.basic_func(x, i)  # 系数 * 基函数
        num_sol = self.cal_numerical_sol(xi[2:-2])
        return num_sol

    def cal_numerical_sol(self, xi):
        """
        计算数值解
        :param xi: 离散数值向量
        :return:
        """
        x = sympy.symbols("x")  # 符号变量
        ux = []  # 针对每个基函数，转化为lambda函数，然后进行数值运算
        for i in range(self.n + 2):
            ux.append(sympy.lambdify(x, self.ux[i], "numpy"))
        num_sol = np.zeros(len(xi))  # 存储数值解
        for i in range(self.n + 2):
            num_sol += ux[i](xi)
        return num_sol

    def plt_ode_curve(self):
        """
        可视化ODE数值解
        :return:
        """

        def plt_num_sol(xi, num_sol):
            plt.plot(xi, num_sol, "--", lw=2, label="$Numerical\ solution \ \hat y(x)$")
            plt.xlabel("$x$", fontdict={"fontsize": 18})
            plt.ylabel("$y(x) \ / \ \hat y(x)$", fontdict={"fontsize": 18})
            plt.title("$B$样条基函数有限元法求解二阶$ODE$数值解", fontdict={"fontsize": 18})
            plt.legend(frameon=False, fontsize=16)
            plt.grid(ls=":")
            plt.tick_params(labelsize=18)  # 刻度字体大小16

        xi = np.linspace(self.a, self.b, 100)
        num_sol = self.cal_numerical_sol(xi)
        if self.ode_model is None:
            plt.figure(figsize=(7, 5))
            plt_num_sol(xi, num_sol)
            plt.show()
        else:
            analytical_sol = self.ode_model(xi)  # 解析解
            plt.figure(figsize=(14, 5))
            plt.subplot(121)
            plt.plot(xi, analytical_sol, "-", lw=2, label="$Analytical \ solution \ y(x)$")
            plt_num_sol(xi, num_sol)
            plt.subplot(122)
            error = analytical_sol - num_sol
            error_norm = np.linalg.norm(error)
            plt.plot(xi, error, "-", lw=1.5)
            plt.xlabel("$x$", fontdict={"fontsize": 18})
            plt.ylabel("$\epsilon = y_k - \hat y_k$", fontdict={"fontsize": 18})
            plt.title("数值解的误差曲线$\Vert y - \hat{y} \Vert_2=%.5e$" % error_norm, fontsize=18)
            plt.grid(ls=":")
            plt.tick_params(labelsize=18)  # 刻度字体大小16
            plt.show()
