# -*- coding: UTF-8 -*-
"""
@file:best_square_approximation.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import sympy
import numpy as np
import math
from scipy import integrate
from function_approximation_03.utils.best_approximation_utils import BestApproximationUtils
from iterative_solution_linear_equation_07.pre_conjugate_gradient \
    import PreConjugateGradient  # 预处理共轭梯度法


class BestSquarePolynomiaApproximation(BestApproximationUtils):
    """
    最佳平方多项式逼近，继承BestApproximationUtils的属性和方法
    """

    def fit_approximation(self):
        """
        最佳平方逼近核心算法
        :return:
        """
        t = self.fun.free_symbols.pop()  # 获取方程的自由变量符号
        H = np.zeros((self.k + 1, self.k + 1), dtype=np.float64)
        d = np.zeros(self.k + 1)
        func = self.fun / t  # 初始化，方便循环内统一
        for i in range(self.k + 1):
            # H矩阵的第一行
            H[0, i] = (math.pow(self.b, i + 1) - math.pow(self.a, i + 1)) / (i + 1)
            func = func * t  # 被积函数，随着i增加，累乘，幂次增加
            expr = sympy.lambdify(t, func)
            d[i] = integrate.quad(expr, self.a, self.b, full_output=1)[0]  # 数值积分
        for i in range(1, self.k + 1):
            # H上一行从第二个元素开始赋值给下一行，从第一个元素到倒数第二个元素
            H[i, :-1] = H[i - 1, 1:]
            # 计算H当前行的最后一个元素
            f1, f2 = math.pow(self.b, self.k + 1 + i), math.pow(self.a, self.k + 1 + i)
            H[i, -1] = (f1 - f2) / (self.k + i + 1)  # 形成H矩阵当前行的最后一个值
        self.poly_coefficient = np.linalg.solve(H, d)  # 求解逼近多项式的系数
        # pre_cg = PreconditionedConjugateGradient(H, d, np.zeros(len(d)), eps=1e-16)
        # self.poly_coefficient = pre_cg.fit_solve()

        # 逼近多项式各项特征组合
        px = sympy.Matrix.zeros(self.k + 1, 1)
        for i in range(self.k + 1):
            px[i] = np.power(t, i)  # p(x)多项式
        self.approximation_poly = self.poly_coefficient[0] * px[0]  # 符号运算
        for i in range(1, self.k + 1):
            self.approximation_poly += self.poly_coefficient[i] * px[i]
        polynomial = sympy.Poly(self.approximation_poly, t)
        self.polynomial_orders = polynomial.monoms()[::-1]  # 阶次，从低到高
        BestApproximationUtils.error_analysis(self)  # 调用父类方法

    def predict_x0(self, x0):
        """
        求解逼近多项式给定点的值
        :return:
        """
        return BestApproximationUtils.predict_x0(self, x0)

    def plt_approximate(self, is_show=True, is_fh_marker=False):
        """
        可视化函数和逼近多项式函数
        :return:
        """
        BestApproximationUtils.plt_approximation(self, "最佳平方", is_show, is_fh_marker)
