# -*- coding: UTF-8 -*-
"""
@file_name: cubic_hermite_2_order_differentiation.py
@time: 2021-11-25
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from numerical_differentiation_05.discrete_data_cubic_hermite_differetiation \
    import DiscreteDataCubicHermiteDifferential  # 三次埃尔米特函数求解一阶数值微分类


class CubicHermite2OrderDifferentiation(DiscreteDataCubicHermiteDifferential):
    """
    三次样条插值求解二阶数值微分，继承DiscreteDataCubicHermiteDifferential
    """

    def cal_diff(self, x0):
        """
        三次埃尔米特方法求解二阶数值微分核心算法，重写父类实例方法
        :return:
        """
        x0 = np.asarray(x0, dtype=np.float64)  # 求微分点
        self.diff_value = np.zeros(len(x0))  # 存储微分值
        # 如果使用第一种边界条件，则根据五点公式构造边界条件的一阶导数
        h = np.mean(np.diff(self.x[:5]))  # 前五个点步长均值
        y_0 = np.dot(np.array([-25, 48, -36, 16, -3]), self.y[:5]) / (12 * h)  # 左端点
        h = np.mean(np.diff(self.x[-5:]))  # 后五个点步长均值
        y_n = np.dot(np.array([3, -16, 36, -48, 25]), self.y[-5:]) / (12 * h)  # 右端点
        mat = self._cal_complete_spline_(y_0, y_n)  # 求解样条系数，第一种条件
        # mat = self._cal_natural_spline_()  # 求解样条系数， 自然样条条件
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
            # 分段三次样条插值二阶导函数公式
            lj, lj_1 = (x0[i] - self.x[idx]) / h, (x0[i] - self.x[idx + 1]) / h
            self.diff_value[i] = 6 * (lj + lj_1) * (self.y[idx] - self.y[idx + 1]) / \
                                 h ** 2 + 2 * ((2 * lj_1 + lj) * mat[idx] +
                                               (2 * lj + lj_1) * mat[idx + 1]) / h
        return self.diff_value
