# -*- coding: UTF-8 -*-
"""
@file_name: simulate_anneal.py
@time: 2022-08-16
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np


class SimulatedAnnealingOptimization:
    """
    模拟退火算法，求解函数的最优值，默认最小值，最大值则需在目标函数前添加减号
    """

    def __init__(self, func, args_span, eps=1e-16, epochs=100, T0=100,
                 Tf=1e-8, alpha=0.98):
        self.func = func  # 待优化的函数
        self.args_span = np.asarray(args_span)  # 求解区间，格式为[[x0, xn], [y0, yn], [z0, zn], ...]
        self.eps = eps  # 精度控制
        self.epochs = epochs  # 内循环迭代次数
        self.alpha = alpha  # 降温系数，越接近于1，优化过程越慢
        self.T = T0  # 初始温度，默认100，该值根据系数alpha不断变化，表示当前温度状态
        self.Tf = Tf  # 温度终值，默认1e-8
        self.n_args = self.args_span.shape[0]  # 参数变量个数
        self.best_y_optlist = []  # 模拟退火中目标值的寻优过程

    def _generate_new_solution(self, x_cur):
        """
        产生新解，并根据当前温度增加扰动，以便调出局部最优解
        :param x_cur: 当前解
        :return:
        """
        yi = np.random.randn(self.n_args)  # 新解的产生，增加扰动
        zi = yi / np.sqrt(sum(yi ** 2))  # 变换
        x_new = x_cur + self.T * zi  # 根据当前温度产生新解
        for k in range(self.n_args):  # 针对每个新解判断范围，使得在求解区间内
            if x_new[k] <= self.args_span[k, 0]:  # 小于左边界
                r = np.random.rand(1)  # [0, 1]均匀分布随机数
                x_new[k] = r * self.args_span[k, 0] + (1 - r) * x_cur[k]
            elif x_new[k] >= self.args_span[k, 1]:  # 大于有边界
                r = np.random.rand(1)
                x_new[k] = r * self.args_span[k, 1] + (1 - r) * x_cur[k]
        return x_new

    def _metropolis(self, y_cur, y_new):
        """
        Metropolis准则
        :param y_cur: 当前解函数值
        :param y_new: 新解函数值
        :return:
        """
        if y_new < y_cur:  # 新值更优，直接接受
            return True
        else:  # 否则，以概率接受
            p = np.exp(-(y_cur - y_new) / self.T)  # 依概率接受
            return True if np.random.rand(1) < p else False  # p大于[0, 1]之间的一个随机数

    def fit_optimize(self):
        """
        模拟退火算法核心部分
        :return:
        """
        x_cur = np.random.rand(self.n_args)  # 初始解，(0, 1)均匀分布
        for i in range(self.n_args):  # 求每个解的初始解区间范围
            x_cur[i] = self.args_span[i, 0] + x_cur[i] * np.diff(self.args_span[i, :])
        y_cur = self.func(x_cur)  # 初始解的函数值
        x_best, y_best = x_cur, y_cur  # 标记最优解和最小函数值
        f_err = 1  # 相邻两次解的绝对差
        # 模拟退火过程，包含两部分：外循环退火过程，内循环，Metroplis算法
        while self.T > self.Tf and f_err > self.eps:  # 外循环，退温过程，根据降温系数不断降温，直到最终温度
            for i in range(self.epochs):  # 内循环，Metroplis算法，每次降温选择最优值
                x_new = self._generate_new_solution(x_cur)  # 根据当前解，产生新解
                y_new = self.func(x_new)  # 生成新解函数值
                # 搜索，当满足优化目标更新；不满足则以概率接受
                if self._metropolis(y_cur, y_new):  # 是否接受新解
                    x_cur, y_cur = x_new, y_new  # 满足Metroplis条件，则接受新解
                # 与目标值对比，选择最优
                if y_cur < y_best:
                    y_best, x_best = y_cur, x_cur
            self.best_y_optlist.append(y_best)  # 存储一轮降温过程中的最优解
            if len(self.best_y_optlist) > 50:  # 降温50轮后计算精度
                f_err = np.abs(self.best_y_optlist[-1] - self.best_y_optlist[-2])
                # err = np.max(np.abs(np.diff(self.best_y_optlist[-51:])))  # 函数最大改变量
                # if err < self.eps:
                #     break
            self.T *= self.alpha  # 更新当前温度
        return [y_best, x_best]
