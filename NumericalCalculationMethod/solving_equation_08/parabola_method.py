# -*- coding: UTF-8 -*-
"""
@file:parabola_method.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import pandas as pd
import datetime


class ThreePointParabolaMethod:
    """
    抛物线法求解方程的根，采用平行弦截法计算初始启动的三个值
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
        抛物线法求解方程的根，核心算法
        :return:
        """
        fa_val, fb_val = self.fx(self.a), self.fx(self.b)  # 左端点，右端点
        # 如果端点处满足精度要求，即为根
        if np.abs(fa_val) <= self.eps:
            self.root = fa_val
            return
        elif np.abs(fb_val) <= self.eps:
            self.root = fb_val
            return
        # 采用平行弦截法启动抛物线法的三个初始起点
        xk_2 = (self.b + self.a) / 2  # 第1个值
        xk_1 = xk_2 - (self.b - self.a) / (fb_val - fa_val) * self.fx(xk_2)  # 第2个值
        xk = xk_1 - (self.b - self.a) / (fb_val - fa_val) * self.fx(xk_1)  # 第3个值
        iter_, fxk = 0, self.fx(xk)  # 初始化迭代变量，以当前更新的值作为精度判断标准
        if np.abs(fxk) <= self.eps:  # 采用平行弦截法确定的初始启动值满足精度要求
            self.root = fxk
            return self.root
        while np.abs(fxk) > self.eps and iter_ < self.max_iter:
            fxk_2, fxk_1 = self.fx(xk_2), self.fx(xk_1)  # 函数值f(x_{k-2})和f(x_{k-1})
            if abs(xk - xk_1) < self.eps or abs(xk_1 - xk_2) < self.eps or \
                    abs(xk - xk_2) < self.eps:
                break
            dq1_nb = (fxk - fxk_1) / (xk - xk_1)  # 1阶差商
            dq1_b0 = (fxk_1 - fxk_2) / (xk_1 - xk_2)  # 1阶差商
            dq2 = (dq1_nb - dq1_b0) / (xk - xk_2)  # 2阶差商
            omega_k = dq1_nb + dq2 * (xk - xk_1)  # wk
            xk_2, xk_1 = xk_1, xk  # 解的迭代
            # 计算新的下一次迭代值
            tmp_val = omega_k ** 2 - 4 * fxk * dq2  # 抛物线公式分母中根号下的值
            if tmp_val < 0:  # np.sqrt(.)不能为小于0
                print("请缩小求解区间.")
                return
            xk = xk - 2 * fxk / (omega_k + np.sign(omega_k) * np.sqrt(tmp_val))  # 抛物线公式
            fxk, iter_ = self.fx(xk), iter_ + 1  # 更新精度，迭代次数+1
            self.root_precision_info.append([iter_, xk, fxk])
        if self.root_precision_info != []:
            self.root_precision_info = np.asarray(self.root_precision_info)  # 便于索引取值操作
            self.root = self.root_precision_info[-1, 1]  # 满足精度的根
            self._display_csv_info()  # 显示信息或存储外部文件
            return self.root

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
