# -*- coding: UTF-8 -*-
"""
@file:nlinequs_fixed_point.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import matplotlib.pyplot as plt
from nonlinear_equations_09.utils.nonlinear_equations_utils import NonLinearEquationsUtils


class NLinearFxFixedPoint(NonLinearEquationsUtils):
    """
    不动点迭代法求解非线性方程组的解，继承NonLinearEquationsUtils，实例化时调用父类__init__()方法
    """
    def  __init__(self, nlin_Fxs, x0, max_iter=200, eps=1e-15, is_plt=False):
        NonLinearEquationsUtils.__init__(self, nlin_Fxs, x0, max_iter, eps, is_plt)
        self.fxs_precision = None  # 最终解向量针对每个方程的精度

    def fit_roots(self):
        """
        不动点迭代法求解非线性方程组的解，核心算法
        :return:
        """
        sol_tol, iter_ = np.min(self.nlin_Fxs(self.x0)), 0  # 初始化精度和迭代变量
        x_n = np.copy(self.x0)  # 注意向量赋值，应为深拷贝copy
        while np.abs(sol_tol) > self.eps and iter_ < self.max_iter:
            x_b = np.copy(x_n)  # 数值解的更新
            x_n = self.nlin_Fxs(x_b)  # 不动点迭代公式，self.nlin_Fxs为迭代函数
            sol_tol, iter_ = np.linalg.norm(self.nlin_Fxs(x_n) - x_n), iter_ + 1  # 更新精度和迭代次数
            self.iter_roots_precision.append([iter_, x_n.flatten(), sol_tol])  # 存储
        self.roots = self.iter_roots_precision[-1][1]  # 满足精度的根
        # 最终解向量针对每个方程的精度
        self.fxs_precision = self.nlin_Fxs(self.roots.reshape(-1, 1)).flatten()
        if self.is_plt:  # 是否可视化图像
            plt.figure(figsize=(14, 5))
            plt.subplot(121)
            self.plt_precision_convergence_curve(False, "Fixed-Point")  # 调用父类方法
            plt.subplot(122)
            self.plt_roots_convergence_curve(False, "Fixed-Point")  # 调用父类方法
            plt.show()
        return self.roots, self.fxs_precision
