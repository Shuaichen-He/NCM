# -*- coding: UTF-8 -*-
"""
@file:iterative_solution_method.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import pandas as pd
import datetime


class IterativeSolutionMethod_Root:
    """
    迭代求解方程的根。包括：不动点Stable迭代法，埃特金Aitken加速迭代法，斯特芬森Steffensen加速迭代法
    """

    def __init__(self, fai_x, x0, eps=1e-15, max_iter=200, display="display",
                 method="steffensen"):
        self.fai_x = fai_x  # 构造的迭代公式
        self.x0 = x0  # 迭代初值
        self.eps = eps  # 近似根的精度要求
        self.max_iter = max_iter  # 最大迭代次数
        self.display = display  # 值有to_csv（存储外部文件），display（只显示最终结果）
        self.method = method  # 迭代的方法，默认为steffensen
        self.root_precision_info = []  # 存储近似根、精度等迭代信息
        self.root = None  # 最终的解

    def fit_root(self):
        """
        迭代求解非线性方程的根核心算法
        :return:
        """
        iter_, tol = 0, np.infty  # 迭代次数和精度初始化
        if self.method == "stable":  # 不动点迭代法
            self._stable_iteration(self.x0, tol, iter_)
        elif self.method == "aitken":
            self._aitken_acceleration(self.x0, tol, iter_)
        elif self.method == "steffensen":
            self._steffensen_iteration(self.x0, tol, iter_)
        else:
            raise ValueError("迭代方法只能是stable、aitken和steffensen")
        self.root_precision_info = np.asarray(self.root_precision_info)  # 便于索引取值操作
        self.root = self.root_precision_info[-1, 1]  # 满足精度的根
        self._display_csv_info()  # 显示信息或存储外部文件
        return self.root

    def _stable_iteration(self, x_n, tol, iter_):
        """
        不动点迭代法
        :return:
        """
        while tol > self.eps and iter_ < self.max_iter:
            x_b = x_n  # 解的迭代更新，即下一次x(n+1)计算结果赋值给上一次x(n)
            x_n = self.fai_x(x_b)  # 计算一次迭代公式
            iter_, tol = iter_ + 1, np.abs(x_n - x_b)  # 精度更新，迭代次数+1
            self.root_precision_info.append([iter_, x_n, tol])

    def _aitken_acceleration(self, x_n, tol, iter_):
        """
        埃特金加速法
        :return:
        """
        xk_seq = np.zeros(3)  # 初始维护不动点迭代法的三个值，存储不动点迭代法的迭代序列
        xk_seq[0] = self.x0
        for i in range(2):  # 至少三个点才能修正一次
            xk_seq[i + 1] = self.fai_x(xk_seq[i])  # 存储迭代序列
        while tol > self.eps and iter_ < self.max_iter:
            x_b = x_n  # 加速的值更新
            x_n = xk_seq[0] - (xk_seq[1] - xk_seq[0]) ** 2 / \
                  (xk_seq[2] - 2 * xk_seq[1] + xk_seq[0])  # 作一次修正
            # xk_seq[2] = x_n  # 为进一步提供加速，修正的值替换最后一个值
            xk_seq[:2] = xk_seq[1:]  # 替换不动点的迭代序列，后两个值赋值给前两个值
            xk_seq[2] = self.fai_x(xk_seq[-1])  # 不动点迭代法计算一次序列，第三个值
            iter_, tol = iter_ + 1, np.abs(x_n - x_b)  # 精度更新，迭代次数+1
            self.root_precision_info.append([iter_, x_n, tol])

    def _steffensen_iteration(self, x_n, tol, iter_):
        """
        斯特芬森迭代法
        :return:
        """
        while tol > self.eps and iter_ < self.max_iter:
            x_b = x_n  # 解的迭代更新，即下一次x(n+1)计算结果赋值给上一次x(n)
            y_n = self.fai_x(x_b)  # 通过迭代公式计算
            z_n = self.fai_x(y_n)  # 加速一次
            if np.abs(z_n - y_n) < self.eps:  # 新计算出来的值已经满足精度
                x_n = z_n
            else:
                x_n = x_b - (y_n - x_b) ** 2 / (z_n - 2 * y_n + x_b)  # 斯特芬森迭代公式
            iter_, tol = iter_ + 1, np.abs(x_n - x_b)  # 精度更新，迭代次数+1
            self.root_precision_info.append([iter_, x_n, tol])

    def _display_csv_info(self):
        """
        求解过程的显示控制，以及把迭代信息存储到外部文件
        :return:
        """
        if self.display.lower() == "to_csv":
            res = pd.DataFrame(self.root_precision_info, columns=["n_iter", "root", "precision"])
            res.to_csv("../result_file/result%s.csv" % datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
        elif self.display.lower() == "display":  # 显示
            info = self.root_precision_info[-1, :]  # 最终的信息
            print(self.method, "Iter：%d, x = %.25f, Precision：%.25e" % (info[0], info[1], info[2]))
