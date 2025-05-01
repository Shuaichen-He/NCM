# -*- coding: UTF-8 -*-
"""
@file_name: discrete_data_cubic_hermite_differetiation.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from util_font import *


class DiscreteDataCubicHermiteDifferential:
    """
    三次埃尔米特方法求解数值微分：仅实现第一种边界条件和自然边界条件，
    第一种边界条件使用五点公式求解边界处一阶导数，且离散数据应为等距
    可根据第2章三次样条插值实现其他边界条件系数的求解。
    """
    diff_value = None  # 存储给定点x0的微分值

    def __init__(self, x, y, boundary_cond="natural"):
        self.x, self.y = np.asarray(x), np.asarray(y)  # 离散数据
        if len(self.x) == len(self.y) and len(self.x) > 5:
            self.n = len(self.x)  # 已知数据点的数量
        else:
            raise ValueError("数据点(x, y)的维度不一致，或节点过少，不适宜采用三次埃尔米特函数求解。")
        self.boundary_cond = boundary_cond  # 边界条件

    def predict_diff_x0(self, x0):
        """
        核心算法：三次埃尔米特方法求解数值微分
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
            mat = self._cal_complete_spline_(y_0, y_n)  # 求解系数，第一种条件
        else:
            mat = self._cal_natural_spline_()  # 求解系数， 自然条件
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
            # 分段三次埃尔米特插值一阶导函数公式
            lj, lj_1 = (x0[i] - self.x[idx]) / h, (x0[i] - self.x[idx + 1]) / h
            dfj = (self.y[idx] - self.y[idx + 1]) / h
            self.diff_value[i] = 6 * lj * lj_1 * dfj + lj_1 * (lj_1 + 2 * lj) * \
                                 mat[idx] + lj * (lj + 2 * lj_1) * mat[idx + 1]
        return self.diff_value

    def _cal_complete_spline_(self, y_0, y_n):
        """
        三次埃尔米特函数：第一种边界条件
        """
        coefficient_matrix = np.diag(2 * np.ones(self.n))  # 求解m的系数矩阵
        b_vector = np.zeros(self.n)
        for i in range(1, self.n - 1):
            u = (self.x[i] - self.x[i - 1]) / (self.x[i + 1] - self.x[i - 1])  # 分母为两个步长和
            lambda_ = (self.x[i + 1] - self.x[i]) / (self.x[i + 1] - self.x[i - 1])
            # 右端向量
            b_vector[i] = 3 * lambda_ * (self.y[i] - self.y[i - 1]) / \
                          (self.x[i] - self.x[i - 1]) + 3 * u * \
                          (self.y[i + 1] - self.y[i]) / (self.x[i + 1] - self.x[i])
            # 形成系数矩阵
            coefficient_matrix[i, i + 1], coefficient_matrix[i, i - 1] = u, lambda_
        b_vector[0], b_vector[-1] = 2 * y_0, 2 * y_n  # 仅仅需要边界两个值
        coefficient = np.linalg.solve(coefficient_matrix, b_vector)  # 求解方程组
        return coefficient

    def _cal_natural_spline_(self):
        """
        求解第二种自然边界条件，边界值处的二阶导数值为0
        :return:
        """
        coefficient_mat = np.diag(2 * np.ones(self.n))  # 求解m的系数矩阵
        coefficient_mat[0, 1], coefficient_mat[-1, -2] = 1, 1
        c_vector = np.zeros(self.n)
        for i in range(1, self.n - 1):
            u = (self.x[i] - self.x[i - 1]) / (self.x[i + 1] - self.x[i - 1])  # 分母为两个步长和
            lambda_ = (self.x[i + 1] - self.x[i]) / (self.x[i + 1] - self.x[i - 1])
            c_vector[i] = 3 * lambda_ * (self.y[i] - self.y[i - 1]) /\
                          (self.x[i] - self.x[i - 1]) + 3 * u * \
                          (self.y[i + 1] - self.y[i]) / (self.x[i + 1] - self.x[i])
            coefficient_mat[i, i + 1], coefficient_mat[i, i - 1] = u, lambda_
        # 仅仅需要边界两个值
        c_vector[0] = 3 * (self.y[1] - self.y[0]) / (self.x[1] - self.x[0])
        c_vector[-1] = 3 * (self.y[-1] - self.y[-2]) / (self.x[-1] - self.x[-2])
        coefficient = np.linalg.solve(coefficient_mat, c_vector)  # 求解方程组
        return coefficient

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
        plt.title("三次埃尔米特插值求解离散数据数值微分", fontdict={"fontsize": 18})
        plt.tick_params(labelsize=16)  # 刻度字体大小16
        plt.grid(ls=":")
        if is_show:
            plt.show()
