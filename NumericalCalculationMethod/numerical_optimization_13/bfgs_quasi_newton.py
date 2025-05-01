# -*- coding: UTF-8 -*-
"""
@file_name: bfgs_quasi_newton.py
@time: 2022-09-14
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from decimal import Decimal  # 精于计算
import sympy
from numerical_optimization_13.golden_section_search import \
    GoldenSectionSearchOptimization  # 黄金分割搜索法
from direct_solution_linear_equations_06.qr_orthogonal_decomposition import \
    QROrthogonalDecomposition  # QR正交分解法求解方程组的解
from util_font import *


class BFGSQuasiNewtonOptimization:
    """
    拟牛顿法，BFGS算法
    """

    def __init__(self, obj_f, x0, B0=None, eps=1e-10, is_minimum=True):
        self.obj_f = obj_f  # 目标优化函数，符号形式，有此计算梯度向量
        self.x0 = np.asarray(x0)  # 初始点
        self.n = len(self.x0)  # n元变量
        self.grad_g = self._cal_grad_fun()  # 计算梯度向量
        if B0 is None:  # 不指定，则默认单位矩阵
            self.B0 = np.eye(self.n)  # 初始单位矩阵，正定对称矩阵
        self.eps = eps  # 精度要求
        self.is_minimum = is_minimum  # 是否是极小值，极大值设置为False
        self.local_extremum = None  # 搜索过程，极值点

    def _cal_grad_fun(self):
        """
        计算多元函数的梯度向量
        :return:
        """
        x = sympy.symbols("x_1:%d" % (self.n + 1))
        grad = sympy.zeros(self.n, 1)
        for i in range(self.n):
            grad[i] = sympy.diff(self.obj_f, x[i])
        return grad

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
            grad_v[i] = self.grad_g[i].subs(x_dict)
        return grad_v

    def fit_optimize(self):
        """
        拟牛顿BFGS法优化多元函数算法的核心内容
        :return:
        """
        x_new, new_Bk, new_gk = self.x0, self.B0, self._cal_grad_val(self.x0)
        local_extremum = []  # 最后一列为函数的极值，初始不包含在内
        t = sympy.symbols("t")  # 表示lambda变量
        x = sympy.symbols("x_1:%d" % (self.n + 1))  # n个符号变量
        err, grad_err = 1, 1
        while err > self.eps and grad_err > 1e-09:
            x_k, gk, Bk = x_new, new_gk, new_Bk  # 最优解、梯度和近似矩阵的的更新
            srd = QROrthogonalDecomposition(Bk, -gk)  # QR正交分解法求解方程组的解
            pk = srd.fit_solve()  # 求解方程组，计算搜索方向
            x_tmp = x_k + t * pk  # 计算新的极值点，并进行一维搜索
            lambda_ = self._search_1d_golden(self.obj_f, t, x, x_tmp)  # 一维搜索
            if lambda_ < 0:  # lambda大于等于0
                lambda_ = 0
            x_new = x_k + lambda_ * pk  # 下一次迭代点x(k+1)
            new_gk = self._cal_grad_val(x_new)
            # 如下求解近似矩阵的更新，new_Gk的精度控制，统一到while中，偷个懒
            delta_k, y_k = (x_new - x_k).reshape(-1, 1), (new_gk - gk).reshape(-1, 1)  # 默认列向量
            P_k_me = np.dot(y_k.T, delta_k)[0, 0]  # Pk的分母，随着逼近，逐渐趋近于0
            P_k = np.dot(y_k, y_k.T) / P_k_me if P_k_me > 1e-50 else 0
            Q_k_me = np.dot(np.dot(delta_k.T, Bk), delta_k)[0, 0]  # Qk的分母，随着逼近，逐渐趋近于0
            Q_k = -1 * np.dot(np.dot(np.dot(Bk, delta_k), delta_k.T), Bk) / Q_k_me \
                if Q_k_me > 1e-50 else 0
            new_Bk = Bk + P_k + Q_k  # BFGS公式
            grad_err = np.linalg.norm(new_gk)  # 梯度的值，一个逐渐减少的值
            # 由于一维搜索逐渐逼近局部最小值，其梯度值不一定非常小，如1e-16，故精度控制方法如下
            err = np.linalg.norm(new_gk - gk)  # 相邻两次梯度值的范数
            local_extremum.append(np.append(x_new, self._cal_fun_val(x_new)))  # 存储当前迭代的最优值
        self.local_extremum = np.asarray(local_extremum)
        if self.is_minimum is False:  # 极大值
            self.local_extremum[:, -1] = -1 * self.local_extremum[:, -1]
        return self.local_extremum[-1]

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
        return self.obj_f.subs(x_dict)  # 求函数值

    def _search_1d_golden(self, f_lambda, t, x, args_x):
        """
        每个一维搜索方向的一元函数，变量为t
        :return:
        """
        for i in range(self.n):
            f_lambda = f_lambda.subs(x[i], args_x[i])
        f_lambda = sympy.lambdify(t, f_lambda, modules="sympy")
        x_span = self._forward_backward(f_lambda)  # 确定单峰区间
        if np.abs(np.diff(x_span)) < 1e-16:  # 单峰区间可能存在过小，极小值可能在原点取得
            return 0.0
        else:
            gss = GoldenSectionSearchOptimization(f_lambda, x_span, 1e-10)  # 黄金分割搜索
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
        plt.title("拟牛顿法$BFGS$优化过程，迭代$%d$次" % len(self.local_extremum), fontdict={"fontsize": 18})
        plt.tick_params(labelsize=16)  # 刻度字体大小16
        plt.grid(ls=":")
        plt.show()