# -*- coding: UTF-8 -*-
"""
@file_name:adaptive_spline_approximation.py
@IDE:PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""

import numpy as np
from function_approximation_03.utils.cubic_spline_interpolation import \
    CubicSplineNaturalInterpolation  # 导入自然样条下的三次样条插值类
from util_font import *


class AdaptiveSplineApproximation:
    """
    自适应样条逼近，节点序列未必等距划分的，每个小区间也并非等长的。
    采用第二种自然边界条件，因为其二阶导数存在且为0。
    如果学习数值微分，可以采用三点公式、五点公式计算边界的一阶导数或二阶导数
    """
    max_error = None  # 逼近精度，最大绝对值误差
    spline_obj = None  # 样条插值的多项式对象
    node_num = 0  # 划分结点数

    def __init__(self, fun, interval, eps=1e-5, max_split_nodes=1000):
        self.fun = fun  # 所逼近的函数
        self.a, self.b = interval[0], interval[1]  # 区间左右端点
        # 自适应划分的节点，样条插值最少三个离散点
        self.node = np.array([self.a, (self.a + self.b) / 2, self.b])
        self.eps, self.max_split_nodes = eps, max_split_nodes  # 逼近精度，以及最大划分节点数

    def fit_approximation(self):
        """
        自适应样条逼近算法
        :return:
        """
        self.max_error, flag, n = 0, True, 10  # 初始化最大误差、循环划分标记、初始分段数
        self.node_num = len(self.node)  # 标记划分区间段节点数
        while flag and len(self.node) <= self.max_split_nodes:
            flag = False
            # 求解划分区间段节点的函数值
            y_node_val = self.fun(np.asarray(self.node))
            k_node, insert_num = np.copy(self.node), 0
            # 选择自然条件，因为其边界处二阶导数为0，故无需提供
            self.spline_obj = CubicSplineNaturalInterpolation(k_node, y_node_val)
            self.spline_obj.fit_interp_natural()  # 建立自然条件三次样条插值多项式
            for i in range(len(k_node) - 1):
                nodes_merge = []  # 用于合并划分的节点
                mx, me = self.find_max_error_x(k_node[i], k_node[i + 1], n)
                # 找到区间上[knode[i], knode[i + 1]]的误差最大的点
                if me > self.eps:
                    # 大于精度，则划分区间为两段，并将此点加入节点数组中
                    nodes_merge.extend(self.node[:i + insert_num + 1])
                    nodes_merge.extend([mx])
                    self.node_num += 1
                    nodes_merge.extend(self.node[i + insert_num + 1:self.node_num - 1])
                    self.node = np.copy(nodes_merge)
                    flag = True
                    insert_num += 1
                elif me > self.max_error:
                    self.max_error = me  # 记录所有分段线性插值区间上的最大误差
        if len(self.node) > self.max_split_nodes:
            print("达到最大划分节点序列数量，最终为：", len(self.node))
        self.node_num = len(self.node)  # 存储划分节点数

    def find_max_error_x(self, s_a, s_b, n):
        """
        找出指定区间中的最大误差和坐标点
        :param s_a、s_b: 指定划分子区间的左、右端点
        :param n: 每次划分的数
        :return:
        """
        eps0 = 1e-2  # 区间误差精度，不易过小
        max_error, max_x = 0, s_a  # 记录区间最大误差和坐标
        tol, max_error_before = 1, 0  # 初始化精度和最大误差所对应的坐标，即x值
        # tol以相邻两次划分节点前后的最大误差绝对差为判断依据
        while tol > eps0:
            if s_b - s_a < self.eps:
                break
            t_n = np.linspace(s_a, s_b, n + 1)  # 划分节点, n段
            p_val = self.spline_obj.predict_x0(t_n)  # 样条插值得出的函数值
            f_val = self.fun(t_n)  # 函数在给定点的值
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
        求解逼近多项式给定点的值，采用三次样条插值求解
        :return:
        """
        return self.spline_obj.predict_x0(x0)

    def plt_approximate(self, is_show=True):  # 参考自适应分段线性逼近
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
        plt.title("自适应分三次样条逼近$(MAE=%.2e)$" % mae, fontdict={"fontsize": 18})
        if is_show:
            plt.show()
