# -*- coding: UTF-8 -*-
"""
@file:b_spline_interpolation.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import sympy
import matplotlib.pyplot as plt
from interpolation_02.utils.piecewise_interp_utils import PiecewiseInterpUtils


class BSplineInterpolation(PiecewiseInterpUtils):
    """
    B样条插值：等距节点三次样条插值。继承PiecewiseInterpUtils父类
    """

    def __init__(self, x, y, dy=None, d2y=None, boundary_cond="natural"):
        PiecewiseInterpUtils.__init__(self, x, y)  # 继承父类方法
        self.dy = np.asarray(dy, dtype=np.float64)  # 边界条件，一阶导数
        self.d2y = np.asarray(d2y, dtype=np.float64)  # 边界条件，二阶导数
        self.boundary_cond = boundary_cond  # 边界条件
        self.h = None

    def fit_interp(self):
        """
        生成B样条插值多项式
        :return:
        """
        self.h = PiecewiseInterpUtils.check_equidistant(self)  # 判断是否等距
        t = sympy.Symbol("t")  # 定义符号变量
        self.polynomial = dict()  # 插值多项式
        self.n -= 1  # 离散数据节点的区间数 n - 1
        self.poly_coefficient = np.zeros((self.n, 4))
        if self.boundary_cond == "complete":
            if self.dy is None:
                raise ValueError("请给出第一种边界条件的一阶导数值.")
            self.dy = np.asarray(self.dy, dtype=np.float64)
            c = self._complete_bspline_()
        elif self.boundary_cond == "second":
            if self.d2y is None:
                raise ValueError("请给出第二种边界条件的二阶导数值.")
            self.d2y = np.asarray(self.d2y, dtype=np.float64)
            c = self._second_bspline_()
        elif self.boundary_cond == "natural":
            c = self._natural_bspline_()
        elif self.boundary_cond == "periodic":
            c = self._periodic_bspline_()
        else:
            raise ValueError("边界条件为complete, second, natural, periodic.")

        # 生成B样条插值多项式
        for i in range(self.n):
            p1 = c[i] * (1 - t) ** 3 / 6
            p2 = c[i + 1] * (3 * t ** 3 - 6 * t ** 2 + 4) / 6
            p3 = c[i + 2] * (-3 * t ** 3 + 3 * t ** 2 + 3 * t + 1) / 6
            p4 = c[i + 3] * t ** 3 / 6
            pi = p1 + p2 + p3 + p4
            self.polynomial[i] = sympy.expand(pi)  # 对插值多项式展开
            polynomial = sympy.Poly(self.polynomial[i], t)
            mon = polynomial.monoms()
            for j in range(len(mon)):
                self.poly_coefficient[i, mon[j][0]] = polynomial.coeffs()[j]

    def _complete_bspline_(self):
        """
        第一种边界条件, 根据边界条件构造矩阵并求解系数
        :return:
        """
        m_coef, b_vector = np.zeros(self.n + 3), np.zeros(self.n + 1)
        coefficient_mat = np.diag(4 * np.ones(self.n + 1))  # 构造对角线元素
        I = np.eye(self.n + 1)  # 构造单位矩阵
        mat_low = np.r_[I[1:, :], np.zeros((1, self.n + 1))]  # 下三角
        mat_up = np.r_[np.zeros((1, self.n + 1)), I[:-1, :]]  # 上三角
        coefficient_mat = coefficient_mat + mat_low + mat_up  # 构造三对角矩阵A
        coefficient_mat[0, 1], coefficient_mat[-1, -2] = 2, 2
        b_vector[1:-1] = 6 * self.y[1:-1]
        b_vector[0] = 6 * self.y[0] + 2 * self.h * self.dy[0]
        b_vector[-1] = 6 * self.y[-1] - 2 * self.h * self.dy[-1]
        # 解方程组，此处可以更改为第6章的追赶法求解
        d_sol = np.reshape(np.linalg.solve(coefficient_mat, b_vector), -1)
        m_coef[1:-1] = d_sol  # 解系数赋值
        m_coef[0] = d_sol[1] - 2 * self.h * self.dy[0]  # 特殊处理
        m_coef[-1] = d_sol[-2] + 2 * self.h * self.dy[-1]  # 特殊处理
        return m_coef

    def _second_bspline_(self):
        """
        第二种边界条件的求解, 根据边界条件构造矩阵并求解系数
        :return:
        """
        m_coef, b_vector = np.zeros(self.n + 3), np.zeros(self.n - 1)
        coefficient_mat = np.diag(4 * np.ones(self.n - 1))  # 构造对角线元素
        I = np.eye(self.n - 1)  # 构造单位矩阵
        mat_low = np.r_[I[1:, :], np.zeros((1, self.n - 1))]  # 下三角
        mat_up = np.r_[np.zeros((1, self.n - 1)), I[:-1, :]]  # 上三角
        coefficient_mat = coefficient_mat + mat_low + mat_up  # 构造三对角矩阵A
        b_vector[1:-1] = 6 * self.y[2:-2]
        b_vector[0] = 6 * self.y[1] - self.y[0] + self.h ** 2 * self.d2y[0] / 6
        b_vector[-1] = 6 * self.y[-2] - self.y[-1] + self.h ** 2 * self.d2y[-1] / 6
        # 解方程组，此处可以更改为第6章的追赶法求解
        d_sol = np.reshape(np.linalg.solve(coefficient_mat, b_vector), -1)
        m_coef[2:-2] = d_sol
        # 如下分别表示：c_0, c_{-1}, c_n, c_{n+1}
        m_coef[1] = self.y[0] - self.h ** 2 * self.d2y[0] / 6
        m_coef[0] = 2 * m_coef[1] - m_coef[2] + self.h ** 2 * self.d2y[0]
        m_coef[-2] = self.y[-1] - self.h ** 2 * self.d2y[-1] / 6
        m_coef[-1] = 2 * m_coef[-2] - m_coef[-3] + self.h ** 2 * self.d2y[-1]
        return m_coef

    def _natural_bspline_(self):
        """
        求解自然边界条件
        :return:
        """
        self.d2y = np.array([0, 0])  # 仅仅需要边界两个值，且为0
        m_coef = self._second_bspline_()
        return m_coef

    def _periodic_bspline_(self):
        """
        求解第三种周期边界条件, 根据边界条件构造矩阵并求解系数
        :return:
        """
        m_coef, b_vector = np.zeros(self.n + 3), np.zeros(self.n)
        coefficient_mat = np.diag(4 * np.ones(self.n))  # 构造对角线元素
        I = np.eye(self.n)  # 构造单位矩阵
        mat_low = np.r_[I[1:, :], np.zeros((1, self.n))]  # 下三角
        mat_up = np.r_[np.zeros((1, self.n)), I[:-1, :]]  # 上三角
        coefficient_mat = coefficient_mat + mat_low + mat_up  # 构造三对角矩阵A
        coefficient_mat[0, -1], coefficient_mat[-1, 0] = 1, 1
        b_vector[:-1] = 6 * self.y[1:-1]
        b_vector[-1] = 6 * self.y[0]
        # 解方程组，此处可以更改为第6章的追赶法求解
        d_sol = np.reshape(np.linalg.solve(coefficient_mat, b_vector), -1)
        m_coef[2:-1] = d_sol
        # 分别表示c0, c_{n−1}, c_{n+1}
        m_coef[1], m_coef[0], m_coef[-1] = d_sol[-1], d_sol[-2], d_sol[0]
        return m_coef

    def predict_x0(self, x0):
        """
        计算插值点x0的插值，由于需要计算t值，故重写父类方法
        :return:
        """
        x0 = np.asarray(x0, dtype=np.float64)  # 类型转换
        y_0 = np.zeros(len(x0))  # 存储x0的插值
        t = sympy.Symbol("t")  # 获取插值多项式的自由符号变量
        # 对每一个插值点x0求解插值
        idx = 0  # 默认第一个多项式
        for i in range(len(x0)):
            # 查找被插值点x0所处的区间段索引idx
            for j in range(1, self.n):
                if self.x[j] <= x0[i] <= self.x[j + 1] or \
                        self.x[j] >= x0[i] >= self.x[j + 1]:
                    idx = j  # 查找到
                    break  # 若查找到，则终止查找
            t_i = (x0[i] - self.x[0]) / self.h - idx  # 区间为[0, 1]
            # 由于计算误差的存在, t_i可能会出现一个很小的负数或一个略大于1的数
            if round(t_i, 5) < 0 or round(t_i, 5) > 1:  # 防止外插
                raise ValueError("所计算的t值不再范围[0, 1].")
            y_0[i] = self.polynomial[idx].evalf(subs={t: t_i})
        return y_0

    def plt_interpolation(self, x0=None, y0=None, fh=None, is_show=True):
        """
        绘制插值多项式和插值点，因为区间转换问题，重写父类方法
        :return:
        """
        if is_show:  # 用于子图绘制，如果当前图形绘制为一子图，则is_show设置为False
            plt.figure(figsize=(7, 5))
        plt.plot(self.x, self.y, "ro", label="$(x_i,y_i)$")  # 离散插值节点
        xi = np.linspace(min(self.x), max(self.x), 200)  # 插值区间内等分200个离散插值节点
        yi_hat = self.predict_x0(xi)  # 求等分点的插值
        plt.plot(xi, yi_hat, "k-", label="$g(x)$曲线")  # 可视化插值多项式
        if x0 is not None and y0 is not None:
            plt.plot(x0, y0, "bs", markersize=6, label="$(x_0, \hat y_0)$")  # 可视化所求插值点
        mse = 0.0  # 均方误差
        if fh is not None:
            plt.plot(xi, fh(xi), "r--", label="$f(x)$曲线")  # 真实函数曲线
            mse = np.mean((fh(xi) - yi_hat) ** 2)  # 均方误差
        plt.legend(frameon=False, fontsize=16)  # 添加图例，并取消外方框
        plt.grid(ls=":")  # 添加主要网格线，且是虚线
        plt.xlabel("$x$", fontdict={"fontsize": 18})  # 横坐标标记，latex修饰
        plt.ylabel("$f(x) \ /\  g(x)$", fontdict={"fontsize": 18})  # 纵坐标标记，latex修饰
        # plt.ylabel("$g(x)$", fontdict={"fontsize": 18})  # 纵坐标标记，latex修饰
        # plt.xlabel("$time$", fontdict={"fontsize": 18})  # 横坐标标记，latex修饰
        # plt.ylabel("$temperature$", fontdict={"fontsize": 18})  # 纵坐标标记，latex修饰
        plt.tick_params(labelsize=16)  # 刻度字体大小16
        if mse != 0.0:
            plt.title("三次均匀$B$样条插值（$%s$）：$MSE=%.5e$"
                      % (self.boundary_cond, mse), fontdict={"fontsize": 18})  # 标题
        else:
            plt.title("三次均匀$B$样条插值（$%s$）曲线及插值节点"
                      % self.boundary_cond, fontdict={"fontsize": 18})  # 标题
        if is_show:
            plt.show()
