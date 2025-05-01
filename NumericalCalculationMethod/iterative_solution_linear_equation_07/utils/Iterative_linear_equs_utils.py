# -*- coding: UTF-8 -*-
"""
@file_name: Iterative_linear_equs_utils.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from util_font import *


class IterativeLinearEquationsUtils:
    """
    解线性方程组的迭代法，实体工具类，不再实现setXX和getXX方法。
    """

    def __init__(self, A, b, x0, eps=1e-8, max_iter=200, is_out_info=False):
        self.A = np.asarray(A, dtype=np.float64)  # 系数矩阵
        self.b = np.asarray(b, dtype=np.float64)  # 右端向量
        self.x0 = np.asarray(x0, dtype=np.float64)  # 迭代初值
        if self.A.shape[0] != self.A.shape[1]:  # 系数矩阵维度判别
            raise ValueError("系数矩阵A不是方阵.")
        else:
            self.n = self.A.shape[0]
        if len(self.b) != self.n or len(self.x0) != self.n:
            raise ValueError("右端向量b或初始向量x0与系数矩阵维度不匹配.")
        else:
            self.b, self.x0 = self.b.reshape(-1), self.x0.reshape(-1)
        self.eps = eps  # 迭代法的精度要求
        self.max_iter = max_iter  # 最大迭代次数
        self.is_out_info = is_out_info  # 是否输出迭代信息
        self.x = None  # 满足精度要求的近似解
        self.precision = []  # 存储每次迭代误差精度
        self.iterative_info = {}  # 组合存储迭代信息

    def _plt_convergence_precision(self, is_show=True, method="", style="o-"):
        """
        可视化迭代解的精度曲线，is_show便于子图绘制，若为子图，则值为False，method为迭代方法，用于title
        """
        if is_show:
            plt.figure(figsize=(7, 5))
        iter_num = self.iterative_info["Iteration_number"]  # 获取迭代次数
        iter_num = np.linspace(1, iter_num, iter_num)  # 等距取值，作为x轴绘图数据
        plt.semilogy(iter_num, self.precision, "%s" % style, lw=2,
                     label="$\epsilon=%.3e, \ k=%d$" %
                           (self.precision[-1], iter_num[-1]))  # 对数坐标
        plt.xlabel("$Iterations(k)$", fontdict={"fontsize": 18})
        plt.ylabel("$Precision(\epsilon)$", fontdict={"fontsize": 18})
        plt.title("$%s$的$\epsilon=\Vert b - Ax^* \Vert _2$的收敛曲线" % method,
                  fontdict={"fontsize": 18})
        plt.legend(frameon=False, fontsize=18)
        plt.tick_params(labelsize=16)  # 刻度字体大小16
        plt.grid(ls=":")
        if is_show:
            plt.show()
