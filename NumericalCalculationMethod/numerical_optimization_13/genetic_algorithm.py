# -*- coding: UTF-8 -*-
"""
@file_name: genetic_algorithm.py
@time: 2022-08-18
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from decimal import Decimal


class GeneticAlgorithmOptimization:
    """
    遗传算法求解函数的最优值， 默认求最大值，最小值则需在目标函数之前添加负号
    """

    def __init__(self, func, args_span, args_precision, pop_size=50, p_crossover=0.5,
                 p_mutation=0.01, min_epochs=50, max_epochs=1000, eps=1e-16,
                 is_Maximum=True):
        self.func = func  # 待优化的函数，适应度
        self.args_span = np.asarray(args_span)  # 求解区间，格式为[[x0, xn], [y0, yn], [z0, zn], ...]
        self.args_precision = np.asarray(args_precision, np.int64)  # 自变量的精度
        self.max_epochs, self.min_epochs = max_epochs, min_epochs  # 迭代最多、最少次数
        self.pop_size = pop_size  # 种群大小
        self.p_crossover = p_crossover  # 交叉算子
        self.p_mutation = p_mutation  # 变异算子
        self.eps = eps  # 精度控制
        self.is_Maximum = is_Maximum  # 优化目标为最大值，False为最小值
        self.n_args = self.args_span.shape[0]  # 参数变量个数
        self.gene_size = self._cal_gene_size()  # 各变量基因大小，数组
        self.sum_gene_size = int(np.sum(self.gene_size))  # 变量的总基因长度，整数
        self.optimizing_best_f_val = []  # 存储最优解

    def _cal_gene_size(self):
        """
        利用自变量的精度和取值范围计算基因数，精度不宜过大，过大可能出现优化失败
        :return:
        """
        gene_k = np.ones(self.n_args, dtype=np.int64)  # 存储每个变量的基因数，整数
        for i in range(self.n_args):
            # 公式：2^(k - 1) <= (U-L) * 10^s <= 2^k - 1，U-L为区间长度，
            # 编码长度k和所需的精度s有关
            gene_k[i] = np.ceil(np.log2(np.diff(self.args_span[i, :]) *
                                        (10 ** int(self.args_precision[i]))))
        return gene_k

    def _init_gene_encoding(self):
        """
        初始化基因编码，二进制编码
        :return:
        """
        return np.random.randint(2, size=(self.pop_size, self.sum_gene_size))  # 0、1编码

    def _binary_to_decimal(self, pop):
        """
        二进制转十进制，变换到决策变量取值区间
        :param pop: 种群基因编码
        :return:
        """
        # X存储每个变量每个个体的数值
        X = np.zeros((self.pop_size, self.n_args))
        for i in range(self.n_args):
            if i == 0:
                pop_ = pop[:, :self.gene_size[0]]  # 第一个变量的编码
            elif i == self.n_args - 1:
                pop_ = pop[:, self.gene_size[self.n_args - 2]:]  # 最后一个变量的编码
            else:  # 2..n-1的每个变量编码
                pop_ = pop[:, self.gene_size[i - 1]: self.gene_size[i]]
            # 公式：x = L + c * delta,  delta = (U - L) / (2^k - 1),
            # c = sum(b(i) * 2^(i - 1)), i = 1...k
            pw2 = 2 ** np.arange(self.gene_size[i])[::-1]  # 对应编码位的幂次值，从高位到底位
            delta = np.diff(self.args_span[i, :]) / \
                    Decimal((2 ** int(self.gene_size[i]) - 1))
            X[:, i] = self.args_span[i, 0] + np.dot(pop_, pw2) * delta
        return X

    def solve(self):
        """
        遗传算法核心代码，在迭代次数内，按照适应度计算、选择、交叉、变异等操作不断优化
        :return:
        """
        best_f_val, best_x = float("-inf"), 0.0  # 初始目标函数的最优值和最优解
        pop = self._init_gene_encoding()  # 初始化基因编码
        # 迭代进化：适应度计算、选择、交叉、变异等操作
        for i in range(self.max_epochs):
            crr_x = self._binary_to_decimal(pop)  # 二进制转十进制
            fitness = self.func(crr_x).flatten()  # 计算适应度，评估函数值，展平为一维数组
            idx = np.argmax(fitness)  # 当前种群的最大值索引下标
            cur_best_x = crr_x[idx]  # 当前最优解
            if fitness[idx] > best_f_val:
                best_f_val, best_x = fitness[idx], cur_best_x  # 当前最优结构
            pop_selected = self._select_operator(pop, fitness)  # 选择算法
            pop = np.vstack((pop_selected, pop[idx]))  # 添加最后一行当前优秀个体
            pop_c = pop.copy()  # parent会被child替换，所以先copy一份pop
            for parent in pop:
                child = self._crossover_operator(parent, pop_c)  # 交叉算法
                child = self._mutate_operator(child)  # 变异算法
                parent[:] = child
            self.optimizing_best_f_val.append(best_f_val)  # 优化过程中的最优值
            # 精度控制，提取终止迭代
            if i + 1 > self.min_epochs:
                # err = np.mean(np.abs(np.diff(self.optimizing_best_f_val[-51:])))  # 函数改变量的绝对均值
                err = np.max(np.abs(np.diff(self.optimizing_best_f_val[-51:])))  # 函数最大改变量
                if err < self.eps:
                    break
        if self.is_Maximum:  # 最大值
            self.optimizing_best_f_val = np.asarray(self.optimizing_best_f_val)
        else:  # 最小值
            self.optimizing_best_f_val = -1 * np.asarray(self.optimizing_best_f_val)
        return best_f_val, best_x

    def _select_operator(self, pop, fitness):
        """
        类似轮盘赌算法，模拟自然选择，适应度越大，则概率越大，越容易被保留，
        此处选择pop_size - 1个，与已经存在最优的一个，共pop_size个
        :param pop: 当前种群
        :param fitness: 函数适应度数组
        :return:
        """
        # 由于轮盘赌算法要求不能有负数，故选择最小负数并转化为其绝对值，则最小负数变为0，其他均为正数
        v1 = np.abs(np.min(fitness)) if np.min(fitness) < 0 else 0
        # 由于分子每个数加上了最小值的绝对值，故共加了v1 * pop_size个每个个体的选择概率
        p = (fitness + v1) / (fitness.sum() + v1 * self.pop_size)
        # 概率大，则被选中的几率就高，重复的个体索引就多
        idx = np.random.choice(np.arange(self.pop_size),
                               size=self.pop_size - 1, replace=True, p=p)
        return pop[idx]

    def _crossover_operator(self, parent, pop):
        """
        模拟交叉，生成新的子代，单点交叉，大概有p_crossover的父代会产生子代
        :param parent: 当前一个较优的亲本父代
        :param pop: # 当前较优的种群
        :return:
        """
        if np.random.rand(1) < self.p_crossover:  # 变异算子概率
            p_idx = np.random.randint(0, self.pop_size, size=1)  # 随机产生另一个待交叉的亲本父代
            cross_point = np.random.randint(0, self.sum_gene_size, size=1)[0]  # 交叉点
            return np.append(parent[:cross_point], pop[p_idx, cross_point:])
        return parent

    def _mutate_operator(self, child):
        """
        模拟变异，在交叉的基础上按照概率p_mutation选择随机变异的位数和变异点
        :param child: 交叉后的一个子代
        :return:
        """
        k = int(np.round(self.sum_gene_size * self.p_mutation))  # 按概率获得变异的位数
        for i in range(k):
            mutate_point = np.random.randint(0, self.sum_gene_size, size=1)[0]  # 变异点
            child[mutate_point] = 1 if child[mutate_point] == 0 else 0  # 0变1，1变0，产生变异
        return child
