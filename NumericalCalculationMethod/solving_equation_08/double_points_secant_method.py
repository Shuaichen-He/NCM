# -*- coding: UTF-8 -*-
"""
@file:double_points_secant_method.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import pandas as pd
import datetime


class DoublePointsSecantMethod:
    """
    双点弦截法和改进的弦截法两种，采用平行弦截法计算初始启动的两个值
    """

    def __init__(self, fx, x_span, eps=1e-15, max_iter=200, display="display"):
        # 略去必要参数实例属性的初始化
        self.fx = fx  # 待求方程
        self.a, self.b = x_span[0], x_span[1]  # 求解区间
        self.eps = eps  # 近似根的精度要求
        self.max_iter = max_iter  # 最大迭代次数
        self.display = display  # 值有to_csv（存储外部文件），display（只显示最终结果）
        self.root_precision_info = []  # 存储划分的区间，近似根，精度
        self.root = None

    def fit_root(self):
        """
        双点弦截法算法
        :return:
        """
        fa_val, fb_val = self.fx(self.a), self.fx(self.b)  # 左右端点函数值
        # 如果端点处满足精度要求，即为根
        if np.abs(fa_val) <= self.eps:
            self.root = fa_val
            return self.root
        elif np.abs(fb_val) <= self.eps:
            self.root = fb_val
            return self.root
        # 双点弦截法，启动需要两个点，采用平行弦截法确定其中一个点
        xk_b = (self.b + self.a) / 2  # 第一个起始点为区间中点x0
        xk = xk_b - (self.b - self.a) / (fb_val - fa_val) * self.fx(xk_b)  # x1
        if np.abs(self.fx(xk)) <= self.eps:  # 采用平行弦截法确定的初始启动值满足精度要求
            self.root = xk
            return self.root
        self._double_secant(xk_b, xk)  # 双点弦截法，此处可扩展其他弦截法
        if self.root_precision_info != []:
            self.root_precision_info = np.asarray(self.root_precision_info)  # 便于索引取值操作
            self.root = self.root_precision_info[-1, 1]  # 满足精度的根
            self._display_csv_info()  # 显示信息或存储外部文件
            return self.root

    def _double_secant(self, xk_b, xk):
        """
        双点弦截法，xk_b和xk为两个启动值
        """
        tol, iter_ = np.infty, 0  # 初始精度和迭代变量
        fk_b, fk = self.fx(xk_b), self.fx(xk)  # 初始的函数值
        while np.abs(tol) > self.eps and iter_ < self.max_iter:  # 在精度要求下迭代求解
            if np.abs(fk - fk_b) < self.eps:  # 防止溢出
                break
            xk_n = xk - (xk - xk_b) / (fk - fk_b) * fk  # 双点弦截法公式
            xk_b, xk, fk_b, fk = xk, xk_n, fk, self.fx(xk_n)  # 近似值和函数值的更新
            tol, iter_ = fk, iter_ + 1  # 更新精度，迭代次数+1
            self.root_precision_info.append([iter_, xk_n, tol])

    def _display_csv_info(self):  # 参考埃特金加速迭代法
        """
        求解过程的显示控制，以及把迭代信息存储到外部文件
        :return:
        """
        if self.display.lower() == "to_csv":
            res = pd.DataFrame(self.root_precision_info, columns=["n_iter", "root", "precision"])
            res.to_csv("../result_file/result%s.csv" % datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
        elif self.display.lower() == "display":  # 显示
            info = self.root_precision_info[-1, :]  # 最终的信息
            print("Iter：%d, x = %.20f, Precision：%.15e" % (info[0], info[1], info[2]))
