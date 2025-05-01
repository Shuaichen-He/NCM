# -*- coding: UTF-8 -*-
"""
@file_name: ant_colony_algorithm.py
@time: 2022-09-15
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np


class AntColonyAlgorithmOptimization:
    """
    蚁群算法，求解函数的最优值，默认最大值，最小值则需在目标函数前添加减号
    """

    def __init__(self, func, args_span, ant_m=100, rho=0.9, tp_c=0.2, step=0.05,
                 eps=1e-16, max_iter=1000, is_Maximum=True):
        self.func = func  # 待优化的n元函数
        self.args_span = np.asarray(args_span)  # 求解区间，格式为[[x0, xn], [y0, yn], [z0, zn], ...]
        self.n = self.args_span.shape[0]  # 自变量数
        self.ant_m = ant_m  # 蚂蚁数量
        self.rho = rho  # 信息素挥发因子
        self.tp_c = tp_c  # 转移概率常数
        self.step = step  # 局部搜索步长
        self.eps = eps  # 最优值的精度控制
        self.max_iter = max_iter  # 最大迭代次数
        self.is_Maximum = is_Maximum  # 默认最大值
        self.Tau = np.zeros(self.n)  # 信息素
        self.tp_state = np.zeros(self.ant_m)  # 每轮迭代中蚂蚁的转移概率
        self.optimizing_best_f_val = None  # 迭代优化过程中的最优值

    def fit_optimize(self):
        """
        蚁群算法核心部分，初始解空间，计算转移概率，更新解空间，更新信息素，精度控制
        :return:
        """
        # 随机化蚂蚁空间位置，解空间。方法：随机生成[0, 1]均匀数，并映射到区间(起点 + 随机数 * 区间长度)
        len_span = np.diff(self.args_span).flatten()  # 区间长度
        ant_space = self.args_span[:, 0] + np.random.rand(self.ant_m, self.n) * len_span
        self.Tau = self.func(ant_space)  # 初始化每个蚂蚁的信息素，适应度函数计算
        idx = np.argmax(self.Tau)  # 最大信息素索引下标
        best_sol = ant_space[idx, :]  # 初始化目标函数的最优解，信息素最大的
        trace_optimizing_list = [self.func(ant_space[[idx], :])[0]]  # 每代最优值，当前最优解以及函数最优值
        for iter_ in range(1, self.max_iter + 1):
            # 1. 计算状态转移概率
            tau_best = np.max(self.Tau)  # 信息素的最大值，用于计算状态转移概率
            self.tp_state = (tau_best - self.Tau) / tau_best  # 计算状态转移概率
            # 2. 更新蚂蚁的空间位置，即更新解空间
            lambda_ = 1 / iter_  # 当前迭代次数的倒数
            for i in range(self.ant_m):
                # 2.1 根据转移概率常数，进行局部或全局搜索更新，产生新的解
                if self.tp_state[i] < self.tp_c:  # 局部搜索
                    # 公式：x_新 = x_旧 + r * step * λ, r服从U(-1, 1)
                    ant_new = ant_space[i, :] + (2 * np.random.rand(self.n) - 1) * \
                              self.step * lambda_
                else:  # 全局搜索
                    # 公式：x_新 = x_旧 + r * x_span_length, r服从U(-0.5, 0.5)
                    ant_new = ant_space[i, :] + np.diff(self.args_span).flatten() * \
                              (np.random.rand(1) - 0.5)
                # 2.2 边界吸收方式进行边界条件处理，使得在对应变量区间内搜索
                idx_left = np.argwhere(ant_new < self.args_span[:, 0]).flatten()  # 某变量小于左边界索引
                ant_new[idx_left] = self.args_span[idx_left, 0]  # 修改为对应变量的区间左端点
                idx_right = np.argwhere(ant_new > self.args_span[:, 1]).flatten()  # 某变量大于左边界索引
                ant_new[idx_right] = self.args_span[idx_right, 1]  # 修改为对应变量的区间右端点
                # 2.3 判断蚂蚁是否移动，ant_loc[[i], :]保持二维数组形式
                if self.func(ant_new.reshape(1, -1)) > self.func(ant_space[[i], :]):
                    ant_space[i, :] = ant_new  # 更细解空间
            # 3. 信息素更新，一次性更新
            self.Tau = (1 - self.rho) * self.Tau + self.func(ant_space)
            # 4. 信息存储，精度计算
            idx = np.argmax(self.Tau)  # 当前最优解的索引
            best_sol = ant_space[idx, :]  # 当前解空间的最优解
            trace_optimizing_list.append(self.func(ant_space[[idx], :])[0])  # 存储最优解
            if iter_ > 50:
                err = np.max(np.abs(np.diff(trace_optimizing_list[-51:])))  # 函数值最大改变量
                if err < self.eps:
                    break
        if self.is_Maximum:  # 最大值
            self.optimizing_best_f_val = np.asarray(trace_optimizing_list)
        else:  # 最小值
            self.optimizing_best_f_val = -1 * np.asarray(trace_optimizing_list)
        return best_sol, self.optimizing_best_f_val[-1]
