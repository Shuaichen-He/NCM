# -*- coding: UTF-8 -*-
"""
@file:successive_compression_newton_method.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import sympy
from util_font import *


class SuccessiveCompressionNewton:
    """
    逐次压缩牛顿法求解多项式方程的全部根
    """

    def __init__(self, p_coefficient, eps=1e-12, max_iter=1000):
        # 从高阶到低阶次依次输入多项式系数，若某幂次不存在，则输入0
        self.p_coefficient = np.asarray(p_coefficient, dtype=np.float64)  # 多项式系数向量
        if self.p_coefficient[0] != 1.0:
            self.p_coefficient /= self.p_coefficient[0]  # 最高阶次系数归一
        self.eps, self.max_iter = eps, max_iter  # 近似根的精度，最大迭代次数
        self.n = len(self.p_coefficient) - 1  # 根的个数
        # self.Sup = self._bernoulli_max()  # 伯努利法确定根的上确界
        self.Sup = np.max(np.abs(self.p_coefficient)) + 1  # 确定根的上确界
        self.root = np.zeros(self.n)  # 存储根
        self.precision = np.zeros(self.n)  # 存储根的精度

    def _bernoulli_max(self):
        """
        伯努利法求解多项式方程的根的上确界
        :return:
        """
        y = np.zeros(self.n + 1)  # 初始
        y[-2] = 1
        tol, x_sup = 1, np.infty  # 初始精度和上确界
        while tol > self.eps:
            x_sup_temp = x_sup  # 上确界更新
            y[-1] = np.dot(-self.p_coefficient[1:], y[:-1])  # 最高价次系数不取
            y[:-1] = y[1:]  # 更新
            x_sup = y[-2] / y[-3]  # 更新上确界
            tol = np.abs(x_sup - x_sup_temp)  # 改变量
        return x_sup

    def _newton_root(self, equ_f, t, a, b):
        """
        采用符号运算 + 数值运算，牛顿法求解一个根
        :return: 一个近似根x*和精度f(x*)
        """
        diff_equ = sympy.lambdify(t, equ_f.diff(t, 1))  # 方程的一阶导
        equ_f = sympy.lambdify(t, equ_f)  # 转换为lambda函数，方便运算
        fa, fb = equ_f(a), equ_f(b)  # 区间的左右端点函数值
        # 左右端点函数值满足精度要求，则退出
        if np.abs(fa) < self.eps:
            return a, fa
        if np.abs(fb) < self.eps:
            return b, fb
        # 初值的选择，取两端点导数较大者
        dfa, dfb = diff_equ(a), diff_equ(b)  # 端点处一阶导数值
        x_n = a - fa / dfa if dfa > dfb else b - fb / dfb
        # 在精度要求下，采用牛顿法迭代求解根
        sol_tol, iter_ = equ_f(x_n), 0  # 初始解的精度和迭代次数
        while np.abs(sol_tol) > self.eps and iter_ < self.max_iter:
            x_b = x_n  # 近似根的更新
            x_n = x_b - equ_f(x_b) / diff_equ(x_b)  # 牛顿迭代法
            sol_tol, iter_ = equ_f(x_n), iter_ + 1  # 更新精度和迭代次数
        return x_n, sol_tol

    def fit_root(self):
        """
        核心算法：逐步压缩牛顿法求解多项式形式的方程的全部根
        :return:
        """
        t = sympy.Symbol("t", real=True)  # 逐次构造符号多项式，符号变量
        s = np.power(t, np.linspace(0, self.n, self.n + 1, endpoint=True))  # 幂次项
        equ_f = np.dot(s, self.p_coefficient[::-1])  # 当前多项式，系数与幂次项对应点积
        p_c = self.p_coefficient[1:]  # 最高阶次系数不取
        for k in range(self.n):  # 逐次求解多项式每一个实根
            self.root[k], self.precision[k] = \
                self._newton_root(equ_f, t, -self.Sup, self.Sup)
            b = np.zeros(self.n - k + 1)  # 剩余多项式系数
            b[0] = 1  # 最高次幂项系数为1，按b0=1, bk = ak+x0*b{k-1}计算
            for i in range(1, self.n - k + 1):  # 逐次计算其他系数
                b[i] = p_c[i - 1] + self.root[k] * b[i - 1]  # 更新剩余多项式系数
            m = np.linspace(0, self.n - k - 1, self.n - k, endpoint=True)  # 幂次
            s = np.power(t, m)  # 幂次项，即t ** m
            equ_f = sympy.simplify(np.dot(s, b[self.n - k - 1::-1]))  # 构造多项式Fk(x)
            p_c = b[1:len(b)]  # 截取系数，求解下一个多项式
        return self.root

    def plt_polynomial_root(self):
        """
        可视化多项式及其根
        """
        t = sympy.Symbol("t", real=True)
        s = np.power(t, np.linspace(0, self.n, self.n + 1, endpoint=True))
        equ_f = sympy.lambdify(t, np.dot(s, self.p_coefficient[::-1]), "numpy")
        plt.figure(figsize=(7, 5))
        limit = (max(self.root) - min(self.root)) / 50
        xi = np.linspace(min(self.root) - limit, max(self.root) + limit, 100)
        plt.plot(xi, equ_f(xi), "-", lw=1.5, label="$f(x)$")
        plt.plot(xi, np.zeros(len(xi)), ":")
        root = sorted([round(r, 5) for r in self.root])
        plt.plot(self.root, self.precision, "o", label="$x^*=%s$" % root)
        plt.legend(frameon=False, fontsize=16)
        plt.xlabel(r"$x$", fontdict={"fontsize": 18})
        plt.ylabel(r"$f(x)$", fontdict={"fontsize": 18})
        plt.title("逐次压缩牛顿法求解多项式全部零点", fontdict={"fontsize": 18})
        plt.tick_params(labelsize=18)  # 刻度字体大小16
        plt.grid(ls=":")
        plt.show()
