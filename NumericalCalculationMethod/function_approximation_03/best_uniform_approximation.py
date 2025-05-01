# -*- coding: UTF-8 -*-
"""
@file_name:best_uniform_approximation.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import sympy
import numpy as np
import math
from function_approximation_03.utils.best_approximation_utils import BestApproximationUtils


class BestUniformApproximation(BestApproximationUtils):
    """
    最佳一致多项式逼近，列梅兹算法。继承BestApproximationUtils的属性和实例方法
    由于需要生成逼近多项式，故仍采用符号运算 + 数值运算
    """
    cross_point_group = None  # f(x)-p(x)的交错点组

    def __init__(self, fun, k, interval=np.array([-1, 1]), eps=1e-8, h=1e-3):
        BestApproximationUtils.__init__(self, fun, k, interval)  # 继承父类属性
        self.eps = eps  # 逼近精度
        self.h = h  # 步长，用于查找|p(x)-f(x)|误差最大的x点

    def solve_coefficient(self, x, fx):
        """
        求解逼近多项式的系数向量
        :param x: 点集
        :param fx: 点集的原函数精确值
        :return: 系数
        """
        A = np.zeros((self.k + 2, self.k + 2))  # n + 2个交错点
        for i in range(self.k + 2):
            A[i, :-1] = x[i] ** np.arange(0, self.k + 1)
            A[i, -1] = (-1) ** i
        return np.linalg.solve(A, np.asarray(fx, dtype=np.float64))  # p(x)的初始系数

    def fit_approximation(self):
        """
        逼近核心算法
        :return:
        """
        t = self.fun.free_symbols.pop()
        px = sympy.Matrix.zeros(self.k + 1, 1)
        for i in range(self.k + 1):
            px[i] = np.power(t, i)  # p(x)多项式，幂次多项式
        # 1. 初始化x（n + 1次切比雪夫多项式的交错点组）和f(x)，区间[-1, 1]
        x = np.zeros(self.k + 2)  # 初始化的交错点组
        fx = sympy.Matrix.zeros(self.k + 2, 1)  # 符号矩阵
        for i in range(self.k + 2):
            x[i] = 0.5 * (self.a + self.b + (self.b - self.a) *
                          math.cos(np.pi * (self.k + 1 - i) / (self.k + 1)))
            fx[i] = self.fun.evalf(subs={t: x[i]})

        # 2. 构造矩阵，并求解线性方程组，得到初始的逼近多项式
        self.poly_coefficient = self.solve_coefficient(x, fx)  # p(x)的初始系数
        # 3. 确定新的点集
        u = self.poly_coefficient[-1]  # 算法中的u
        max_t, max_x, tol = 0.0, np.infty, np.infty  # 记录abs(fx - px)取最大值的x，精度初始化为正无穷
        while tol > self.eps:
            xi = self.a  # xi初始化为区间左端点
            # 3.1 此循环找出abs(f(x)-p(x))取最大值的x
            while xi < self.b:
                xi += self.h * (self.b - self.a) / self.k  # 等距划分且递增步长
                px1 = np.asarray(px.evalf(subs={t: xi}), dtype=np.float64)  # 各幂次在xi得值
                pt = np.dot(px1.reshape(-1), self.poly_coefficient[:-1].reshape(-1))  # 逼近多项式的近似值
                ft = self.fun.evalf(subs={t: xi})  # 原函数在xi的精确值
                if np.abs(ft - pt) > max_t:
                    max_x, max_t = xi, np.abs(ft - pt)
            if max_x > self.b:  # 未找到，则确定右端点为最大的x
                max_x = self.b

            # 3.2 确定新点集的三种情况
            if self.a <= max_x <= x[1]:  # 第一种情况
                d1, d2 = self.cal_point_set(t, px, x[1], max_x)
                if d1 * d2 > 0:  # 同号，d1表示f(x1)-p(x1)，d2表示f(max_x)-p(max_x)
                    print("第一种情况：替代x(1)，x(1)_old=%.5f, x(1)_new=%.5f" % (x[0], max_x))
                    x[0] = max_x
            elif x[-2] <= max_x <= self.b:  # 第二种情况
                d1, d2 = self.cal_point_set(t, px, x[-2], max_x)
                if d1 * d2 > 0:
                    print("第二种情况：替代x(n+1)，x(n+1)_old=%.5f, x(n+1)_new=%.5f"
                          % (x[-2], max_x))
                    x[-2] = max_x  # x[-1]为u
            else:  # 第三种情况
                idx_x = None  # 找到max_x所在区间的索引
                for i in range(1, self.k + 1):
                    if x[i] <= max_x <= x[i + 1] or x[i + 1] <= max_x <= x[i]:
                        idx_x = i
                        break
                if idx_x is not None:
                    d1, d2 = self.cal_point_set(t, px, x[idx_x], max_x)
                    if d1 * d2 > 0:
                        print("第三种情况：替代x(i+1)，x(i+1)_old=%.5f, x(i+1)_new=%.5f"
                              % (x[idx_x], max_x))
                        x[idx_x] = max_x
            # 3.3 重新计算f(x)的精确值和逼近多项式系数
            for i in range(self.k + 2):
                fx[i] = self.fun.evalf(subs={t: x[i]})
            self.poly_coefficient = self.solve_coefficient(x, fx)  # 求解系数，更新逼近多项式
            tol = np.abs(self.poly_coefficient[-1] - u)  # 精度更新
            u = self.poly_coefficient[-1]  # u更新

        # 4. 满足精度要求后，逼近多项式各项特征组合
        self.poly_coefficient = self.poly_coefficient[:-1].reshape(-1)  # 多项式系数
        self.approximation_poly = self.poly_coefficient[0] * px[0]
        for i in range(1, self.k + 1):
            self.approximation_poly += self.poly_coefficient[i] * px[i]  # 最佳一致逼近多项式
        self.abs_error = dict({"u": u[0], "tol": tol[0]})  # 逼近误差
        polynomial = sympy.Poly(self.approximation_poly, t)  # 构造多项式对象
        self.polynomial_orders = polynomial.monoms()[::-1]  # 阶次，从低到高，故反转
        BestApproximationUtils.error_analysis(self)  # 调用父类，误差分析
        self.cal_cross_point_group(t, x)  # 计算交错点组

    def cal_cross_point_group(self, t, x):
        """
        计算f(x) - p(x)的交错点组
        :return:
        """
        fun_expr = sympy.lambdify(t, self.fun)  # 构成lambda函数
        poly_expr = sympy.lambdify(t, self.approximation_poly)
        self.cross_point_group = fun_expr(x) - poly_expr(x)

    def cal_point_set(self, t, px, x, max_x):
        """
        计算新的点集
        :param t: 符号变量
        :param px: 符号多项式
        :param x: n + 1个点集
        :param max_x: 取最大值的x
        :return:
        """
        f0, fm = self.fun.evalf(subs={t: x}), self.fun.evalf(subs={t: max_x})
        px1 = np.asarray(px.evalf(subs={t: x}), dtype=np.float64)
        pt = np.dot(px1.reshape(-1), self.poly_coefficient[:-1].reshape(-1))
        pm1 = np.asarray(px.evalf(subs={t: max_x}), dtype=np.float64)
        pm = np.dot(pm1.reshape(-1), self.poly_coefficient[:-1].reshape(-1))
        d1, d2 = f0 - pt, fm - pm
        return d1, d2

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
        BestApproximationUtils.plt_approximation(self, "最佳一致(列梅兹)", is_show, is_fh_marker)
