# -*- coding: UTF-8 -*-
"""
@file_name: gradient_descent.py
@time: 2022-09-12
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from decimal import Decimal  # 精于计算
import sympy
from numerical_optimization_13.golden_section_search import GoldenSectionSearchOptimization
from util_font import *


class GradientDescentOptimization:
    """
    最速梯度下降法，求解n元函数的极值问题
    """

    def __init__(self, fun, x0, eps, is_minimum=True):
        self.fun = fun  # 优化函数，符号函数定义
        self.x0 = np.asarray(x0)  # 初始点
        self.n = len(self.x0)  # n元变量
        self.eps = eps  # 精度要求
        self.is_minimum = is_minimum  # 是否是极小值，极大值设置为False
        self.grad_f = self._cal_grad_vector()  # 计算多元函数梯度向量
        self.local_extremum = None  # 搜索过程，极值点

    def _cal_grad_vector(self):
        """
        计算多元函数的梯度向量
        :return:
        """
        x = sympy.symbols("x_1:%d" % (self.n + 1))
        grad = sympy.zeros(self.n, 1)
        for i in range(self.n):
            grad[i] = sympy.diff(self.fun, x[i])
        return grad

    def fit_optimize(self):
        """
        最速梯度下降优化多元函数算法的核心内容
        :return:
        """
        p, p_val = self.x0, self._cal_fun_val(self.x0)
        x = sympy.symbols("x_1:%d" % (self.n + 1))  # n个符号变量
        t = sympy.symbols("t")  # 表示gamma变量
        err, f_err = 1, 1
        local_extremum = [np.append(p, p_val)]  # 最后一列为函数的极值
        while err > self.eps and f_err > self.eps:
            p0 = p  # 更新极值点
            grad_v = self._cal_grad_val(p)  # 梯度向量的值
            S = -1 / np.linalg.norm(grad_v) * grad_v  # 搜索方向
            p = p0 + t * S
            gamma = self._search_1d_golden(self.fun, t, x, p)  # 一维搜索
            p = p0 + gamma * S  # 下一次迭代点
            p_val = self._cal_fun_val(p)  # 下一次迭代点的函数值
            err = np.linalg.norm(p - p0)  # 精度判断1
            f_err = np.abs((p_val - local_extremum[-1][-1]))  # 精度判断2
            local_extremum.append(np.append(p, p_val))  # 存储当前迭代的最优值
        self.local_extremum = np.asarray(local_extremum)
        if self.is_minimum is False:  # 极大值
            self.local_extremum[:, -1] = -1 * self.local_extremum[:, -1]
        return self.local_extremum[-1]

    def _cal_grad_val(self, x_k):
        """
        计算梯度向量的值
        :param x_k: 给定点（x1,x2, ...xn）
        :return:
        """
        x = sympy.symbols("x_1:%d" % (self.n + 1))  # n个符号变量
        x_dict = dict()  # 符号函数求值，对应替换变量和值的字典
        for i in range(self.n):
            x_dict[x[i]] = x_k[i]  # 格式为{x_i: x_0[i}
        grad_v = np.zeros(self.n)  # 梯度值
        for i in range(self.n):
            grad_v[i] = self.grad_f[i].subs(x_dict)
        return grad_v

    def _cal_fun_val(self, x_p):
        """
        计算符号多元函数值
        :param x_p: 求值点x，n元
        :return:
        """
        x = sympy.symbols("x_1:%d" % (self.n + 1))  # n个符号变量
        x_dict = dict()  # 符号函数求值，对应替换变量和值的字典
        for i in range(self.n):
            x_dict[x[i]] = x_p[i]  # 格式为{x_i: x_p[i}
        return self.fun.subs(x_dict)  # 求函数值

    def _search_1d_golden(self, f_gamma, t, x, args_x):
        """
        每个一维搜索方向的一元函数，变量为t
        :return:
        """
        for i in range(self.n):
            f_gamma = f_gamma.subs(x[i], args_x[i])
        f_gamma = sympy.lambdify(t, f_gamma, modules="sympy")
        x_span = self._forward_backward(f_gamma)  # 确定单峰区间
        if np.abs(np.diff(x_span)) < 1e-16:  # 单峰区间可能存在过小，极小值可能在原点取得
            return 0.0
        else:
            gss = GoldenSectionSearchOptimization(f_gamma, x_span, 1e-10)  # 黄金分割搜索
            gamma = gss.fit_optimize()  # 求解极小值
            return gamma[0]

    @staticmethod
    def _forward_backward(ft):
        """
        进退法确定一元函数的单峰区间
        :return:
        """
        step, n, flag = 0.01, 0, -1  # 步长、幂次增量、标记
        x = np.zeros(3)  # 初始猜测点，对于一元搜索，即3个点
        while ft(x[0]) <= ft(x[1]) or ft(x[1]) >= ft(x[2]):
            x[:-1] = x[1:]
            x[-1] = Decimal(x[-1] + step * 2 ** n)
            n += 1
            if np.abs(x[-1]) > 1e+05:  # 避免区间过大，否则反方向进行
                x, n = np.zeros(3), 0
                step = -0.01  # 反方向搜索
                flag += 1
            if flag == 1:  # 进退各一次
                break
        return [x[0], x[-1]]

    def plt_optimization(self, x_zone, y_zone):
        """
        可视化优化过程
        :param x_zone:  可视化x坐标的区间
        :param y_zone:  可视化y坐标的区间
        :return:
        """
        e_p = self.local_extremum[-1]  # 极值点
        xi, yi = np.linspace(x_zone[0], x_zone[1], 100), np.linspace(y_zone[0], y_zone[1], 100)
        x, y = np.meshgrid(xi, yi)
        fxy = np.zeros((100, 100))
        for i in range(100):
            for j in range(100):
                fxy[i, j] = self._cal_fun_val([x[i, j], y[i, j]])
        fig = plt.figure(figsize=(14, 5))
        ax = fig.add_subplot(121)
        if self.is_minimum:
            c = plt.contour(x, y, fxy, levels=15, cmap=plt.get_cmap("jet"))
            plt.clabel(c, inline=True, fontsize=10)  # 添加等高数值标记
            plt.plot(e_p[0], e_p[1], "ko")
        else:
            c = plt.contour(x, y, -1 * fxy, levels=15, cmap=plt.get_cmap("jet"))
            plt.clabel(c, inline=True, fontsize=10)  # 添加等高数值标记
            plt.plot(e_p[0], e_p[1], "ko")
        ax.set_xlabel("$x$", fontdict={"fontsize": 18})
        ax.set_ylabel("$y$", fontdict={"fontsize": 18})
        plt.title("函数局部极值点$((%.5f, %.5f), %.5f)$" % (e_p[0], e_p[1], e_p[2]), fontdict={"fontsize": 18})
        plt.tick_params(labelsize=16)  # 刻度字体大小16
        plt.grid(ls=":")
        plt.subplot(122)
        plt.plot(np.arange(1, len(self.local_extremum) + 1), self.local_extremum[:, -1], "k*--",
                 markerfacecolor="r", markeredgecolor="r")
        plt.xlabel("搜索次数", fontdict={"fontsize": 18})
        plt.ylabel("$f(x^*, y^*)$", fontdict={"fontsize": 18})
        plt.title("梯度法优化过程，迭代$%d$次" % len(self.local_extremum), fontdict={"fontsize": 18})
        plt.tick_params(labelsize=16)  # 刻度字体大小16
        plt.grid(ls=":")
        plt.show()
