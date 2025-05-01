# -*- coding: UTF-8 -*-
"""
@file_name: discrete_data_cubic_spline_differetiation.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from util_font import *


class DiscreteDataCubicSplineDifferential:
    """
    三次样条方法求解数值微分: 仅实现第一种边界条件和自然边界条件,
    第一种边界条件使用五点公式求解边界处一阶导数, 且离散数据应为等距.
    """

    def __init__(self, x, y, boundary_cond="natural"):
        self.x, self.y = np.asarray(x), np.asarray(y)  # 离散数据
        if len(self.x) == len(self.y) and len(self.x) > 5:
            self.n = len(self.x)  # 已知数据点的数量
        else:
            raise ValueError("x,y的维度不一致，或节点过少。")
        self.boundary_cond = boundary_cond  # 边界条件
        self.diff_value = None  # 存储给定点x0的微分值

    def predict_diff_x0(self, x0):
        """
        核心算法：三次样条方法求解数值微分
        :return:
        """
        x0 = np.asarray(x0, dtype=np.float64)  # 求微分点
        self.diff_value = np.zeros(len(x0))  # 存储微分值
        if self.boundary_cond.lower() == "complete":  # 第一种边界条件
            # 如果使用第一种边界条件，则根据五点公式构造边界条件的一阶导数
            h = np.mean(np.diff(self.x[:5]))  # 前五个点步长均值
            y_0 = np.dot(np.array([-25, 48, -36, 16, -3]), self.y[:5]) / (12 * h)  # 左端点
            h = np.mean(np.diff(self.x[-5:]))  # 后五个点步长均值
            y_n = np.dot(np.array([3, -16, 36, -48, 25]), self.y[-5:]) / (12 * h)  # 右端点
            M = self._cal_complete_spline_(self.x, self.y, y_0, y_n)  # 求解样条系数，第一种条件
        else:
            M = self._cal_natural_spline_(self.x, self.y)  # 求解样条系数， 自然样条条件
        for i in range(len(x0)):  # 逐个求解给定值的微分
            # 查找被插值点x0所处的区间段索引idx
            idx = 0  # 初始为第一个区间，从下一个区间开始搜索
            for j in range(1, self.n - 1):
                if self.x[j] <= x0[i] <= self.x[j + 1] or \
                        self.x[j] >= x0[i] >= self.x[j + 1]:
                    idx = j
                    break
            # 求解x0点的导数值
            h = self.x[idx + 1] - self.x[idx]  # 第idx个区间步长
            # 分段三次样条插值一阶导函数公式
            t1, t2 = x0[i] - self.x[idx], self.x[idx + 1] - x0[i]  # 子项
            self.diff_value[i] = t1 ** 2 / (2 * h) * M[idx + 1] - \
                                 t2 ** 2 / (2 * h) * M[idx] - \
                                 (self.y[idx] - M[idx] * h ** 2 / 6) / h + \
                                 (self.y[idx + 1] - M[idx + 1] * h ** 2 / 6) / h
        return self.diff_value

    def _base_args(self, x, y, d_vector, coefficient_mat):
        """
        针对系数矩阵和右端向量的内点计算
        """
        for i in range(1, self.n - 1):  # 针对内点
            lambda_ = (x[i + 1] - x[i]) / (x[i + 1] - x[i - 1])  # 分母为两个步长和
            u = (x[i] - x[i - 1]) / (x[i + 1] - x[i - 1])  # 分母为两个步长和
            df1 = (y[i] - y[i - 1]) / (x[i] - x[i - 1])  # 一阶差商f[x_i, x_{i-1}]
            df2 = (y[i + 1] - y[i]) / (x[i + 1] - x[i])  # 一阶差商f[x_{i+1}, x_i]
            d_vector[i] = 6 * (df2 - df1) / (x[i + 1] - x[i - 1])  # 右端向量元素
            coefficient_mat[i, i + 1], coefficient_mat[i, i - 1] = lambda_, u
        return d_vector, coefficient_mat

    def _cal_complete_spline_(self, x, y, y_0, y_n):
        """
        三次样条函数：第一种边界条件
        """
        coefficient_mat = np.diag(2 * np.ones(self.n))  # 求解m的系数矩阵
        coefficient_mat[0, 1], coefficient_mat[-1, -2] = 1, 1  # 特殊处理
        d_vector = np.zeros(self.n)  # 右端向量
        # 针对内点计算各参数值并构成右端向量和系数矩阵
        d_vector, coefficient_mat = self._base_args(x, y, d_vector, coefficient_mat)
        # 特殊处理两个边界值
        d_vector[0] = 6 * ((y[1] - y[0]) / (x[1] - x[0]) - y_0) / (x[1] - x[0])
        d_vector[-1] = 6 * (y_n - (y[-1] - y[-2]) / (x[-1] - x[-2])) / \
                       (x[-1] - x[-2])
        return np.reshape(np.linalg.solve(coefficient_mat, d_vector), -1)

    def _cal_natural_spline_(self, x, y):
        """
        求解第二种自然边界条件，d2y为边界值处的二阶导数值为0
        :return:
        """
        d2y = np.array([0, 0])  # 边界处的二阶导数值为0
        coefficient_mat = np.diag(2 * np.ones(self.n))  # 求解m的系数矩阵
        # coefficient_mat[0, 1], coefficient_mat[-1, -2] = 0, 0  # 特殊处理
        d_vector = np.zeros(self.n)  # 右端向量
        # 针对内点计算各参数值并构成右端向量和系数矩阵
        d_vector, coefficient_mat = self._base_args(x, y, d_vector, coefficient_mat)
        d_vector[0], d_vector[-1] = 2 * d2y[0], 2 * d2y[-1]  # 仅需边界两个值
        return np.reshape(np.linalg.solve(coefficient_mat, d_vector), -1)

    def plt_differentiation(self, interval, dfh=None, x0=None, y0=None, is_show=True, is_fh_marker=False):
        """
        可视化，随机化指定区间微分节点
        :param is_fh_marker: 真实函数是曲线类型还是marker类型
        :return:
        """
        xi = np.linspace(interval[0], interval[1], 200)  # 等距划分
        y_true = dfh(xi)  # 原函数一阶导函数值
        y_diff = self.predict_diff_x0(xi)  # 三次样条插值求解离散数据数值微分
        # 可视化
        if is_show:
            plt.figure(figsize=(7, 5))
        mae = np.mean(np.abs(y_true - y_diff))
        print("最大绝对值误差：%.10e" % np.max(np.abs(y_true - y_diff)))
        print("平均绝对值误差：%.10e" % mae)
        plt.plot(xi, y_diff, "r-", lw=2, label="数值微分$(MAE=%.2e)$" % mae)
        if is_fh_marker:
            xi = interval[0] + np.random.rand(50) * (interval[1] - interval[0])
            xi = np.array(sorted(xi))  # list-->ndarray，升序排列
            y_true_ = dfh(xi)
            plt.plot(xi, y_true_, "k*", label="$f^{\prime}(x_k), \ x_k \sim U(a, b)$")
        else:
            plt.plot(xi, y_true, "k--", lw=2, label="$f^{\prime}(x)$")
        if x0 is not None and y0 is not None:
            plt.plot(x0, y0, "bo", label="$(x_i, \hat y_i^{\prime})$")
        plt.legend(frameon=False, fontsize=18)
        plt.xlabel(r"$x$", fontdict={"fontsize": 20})
        plt.ylabel(r"$f^{\prime}(x)$", fontdict={"fontsize": 20})
        plt.title("三次样条插值求解离散数据数值微分", fontdict={"fontsize": 18})
        plt.tick_params(labelsize=16)  # 刻度字体大小16
        plt.grid(ls=":")
        if is_show:
            plt.show()
