# -*- coding: UTF-8 -*-
"""
@file_name:adaptive_piecewise_linear_approximation.py
@IDE:PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from util_font import *


class AdaptivePiecewiseLinearApproximation:
    """
    自适应分段线性逼近
    """
    max_error = None  # 自适应分段线性逼近的最大逼近误差
    node_num = 0  # 划分结点数

    def __init__(self, fun, interval, eps=1e-8, max_split_nodes=1000):
        self.fun = fun  # 所逼近的函数
        self.a, self.b = interval[0], interval[1]  # 区间左右端点
        self.node = [self.a, self.b]  # 自适应划分的节点
        self.eps, self.max_split_nodes = eps, max_split_nodes  # 逼近精度，以及最大划分节点数

    def fit_approximation(self):
        """
        自适应逼近算法
        :return:
        """
        self.max_error, flag, n = 0, True, 10  # 初始化最大误差、循环划分标记、初始分段数
        num = len(self.node)  # 标记划分区间段节点数，最大划分为1000，否则退出循环。
        while flag and len(self.node) <= self.max_split_nodes:
            # insert_num表示插入新节点前已插入的结点数
            flag, k_node, insert_num = False, np.copy(self.node), 0
            for i in range(len(k_node) - 1):
                node_join = []  # 用于合并划分的节点
                mx, me = self._find_max_error_x(k_node[i], k_node[i + 1], n)
                # 找到区间上[knode[i], knode[i + 1]]的误差最大的点
                if me > self.eps:
                    # 大于精度，则划分区间为两段
                    node_join.extend(self.node[:i + insert_num + 1])  # 当前插入节点的之前的节点
                    node_join.extend([mx])  # 当前插入节点
                    num += 1  # 节点数增一
                    node_join.extend(self.node[i + insert_num + 1:num - 1])  # 当前插入节点的之后的节点
                    self.node = np.copy(node_join)  # 节点序列更新
                    flag = True  # 已插入节点，故仍需划分
                    insert_num += 1  # 插入节点数增一
                elif me > self.max_error:
                    self.max_error = me  # 记录所有分段线性插值区间上的最大误差
        if len(self.node) > self.max_split_nodes:
            print("达到最大划分节点序列数量，最终为：", len(self.node))
        self.node_num = len(self.node)  # 存储划分节点数

    def _find_max_error_x(self, a, b, n):
        """
        找出指定区间中的最大误差和坐标点
        :param a, b: 指定区间左、右端点
        :param n: 每次划分的数
        :return:
        """
        eps0 = 1e-2  # 区间误差精度，不易过小
        max_error, max_x = 0, a  # 记录区间最大误差和坐标
        fa, fb = self.fun(a), self.fun(b)  # 端点函数值
        tol, max_error_before = 1, 0  # 初始化精度和上一次最大误差
        # tol以相邻两次划分节点前后的最大误差绝对差为判断依据
        while tol > eps0:
            if b - a < self.eps:
                break
            j = np.linspace(0, n, n + 1)  # 划分节点索引下标
            t_n = a + j * (b - a) / n  # 等分的区间点，向量
            p_val = fa + (t_n - a) * (fb - fa) / (b - a)  # 线性插值得出的函数值，向量
            f_val = self.fun(t_n)  # 函数在给定点的值，向量
            error = np.abs(f_val - p_val)  # 求解误差
            max_idx = np.argmax(error)  # 最大误差所对应的索引
            if error[max_idx] > max_error:
                max_error = error[max_idx]  # 记录最大误差
                max_x = t_n[max_idx]  # 记录此点坐标
            tol = np.abs(max_error - max_error_before)  # 更新误差
            max_error_before, n = max_error, n * 2  # 划分节点数增加一倍
        return max_x, max_error

    def predict_x0(self, x0):
        """
        求解逼近多项式给定点的值
        :return:
        """
        y0, idx = np.zeros(len(x0)), 0
        for i in range(len(x0)):
            # 针对每个逼近点x0查找所在区间段
            for j in range(len(self.node) - 1):
                if self.node[j] <= x0[i] < self.node[j + 1] or \
                        self.node[j + 1] <= x0[i] < self.node[j]:
                    idx = j
                    break
            # 构造线性表达式并求解
            y_idx1, y_idx2 = self.fun(self.node[idx]), self.fun(self.node[idx + 1])
            y0[i] = y_idx1 + (y_idx2 - y_idx1) * (x0[i] - self.node[idx]) / \
                    (self.node[idx + 1] - self.node[idx])
        return y0

    def plt_approximate(self, is_show=True):
        """
        可视化图像，逼近曲线为等分200个，离散点为随机50个
        :return:
        """
        if is_show:
            plt.figure(figsize=(7, 5))
        x = np.linspace(self.a, self.b, 200)
        y = self.fun(x)  # 真值
        y_hat = self.predict_x0(x)  # 预测值
        plt.plot(x, y, "k-", lw=1.5, label="$f(x)$")
        plt.plot(x, y_hat, "r--", lw=1.5, label="$p(x)$")
        xi = self.a + np.random.rand(50) * (self.b - self.a)
        xi = np.array(sorted(xi))  # list-->ndarray，升序排列
        yi = self.fun(xi)  # 计算原逼近函数在xi处的值
        plt.plot(xi, yi, "ko", label="$(x_k, y_k)$")
        y0 = self.predict_x0(xi)
        plt.plot(xi, y0, "r*", label="$(x_k, \hat y_k)$")
        plt.legend(frameon=False, fontsize=18)
        plt.grid(ls=":")
        plt.xlabel("$x$", fontdict={"fontsize": 18})
        plt.ylabel(r"$f(x) \ / \ p(x)$", fontdict={"fontsize": 18})
        plt.tick_params(labelsize=16)  # 刻度字体大小16
        mae = np.mean(np.abs(y - y_hat))
        print("平均绝对值误差：%.10e" % mae)
        print("最大绝对值误差：%.10e" % np.max(np.abs(y - y_hat)))
        plt.title("自适应分段线性逼近$(MAE=%.2e)$" % mae, fontdict={"fontsize": 18})
        if is_show:
            plt.show()
