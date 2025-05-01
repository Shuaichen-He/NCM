# -*- coding: UTF-8 -*-
"""
@file:newton_root_method.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import sympy
import pandas as pd
import datetime


class NewtonRootMethod:
    """
    牛顿法求解方程的根，包含牛顿法newton, 牛顿加速哈利法halley, 牛顿下山法downhill
    和重根情形multroot.
    """

    def __init__(self, fx, x0, eps=1e-15, max_iter=200, display="display",
                 method="newton"):
        # 待求根方程转化为lambda函数，以及方程的一阶导和二阶导（针对重根）
        self.fx, self.dfx, self.d2fx = self._solve_diff_fun_(fx)
        self.x0 = x0  # 初值附近的根
        self.eps = eps  # 近似根的精度要求
        self.max_iter = max_iter  # 最大迭代次数
        self.display = display  # 值有to_csv（存储外部文件），display（只显示最终结果）
        self.method = method  # 牛顿迭代法的方法，默认采用经典牛顿法
        self.root_precision_info = []  # 迭代过程信息存储
        self.root = None  # 最终的解

    @staticmethod
    def _solve_diff_fun_(equ):
        """
        求解方程的一阶导数和二阶导数，并把符号函数转换为lanmbda函数
        :return:
        """
        t = equ.free_symbols.pop()  # 获得符号自由变量
        diff_equ = sympy.lambdify(t, equ.diff(t, 1))  # 一阶导
        diff2_equ = sympy.lambdify(t, equ.diff(t, 2))  # 二阶导
        equ_expr = sympy.lambdify(t, equ)  # 原方程转换
        return equ_expr, diff_equ, diff2_equ

    def fit_root(self):
        """
        牛顿法求解方程的根
        :return:
        """
        if self.method == "newton":  # 牛顿法
            self._newton_()
        elif self.method == "halley":  # 哈利加速法
            self._newton_halley_()
        elif self.method == "simple":  # 简单牛顿法
            self._simple_newton_()
        elif self.method == "downhill":  # 牛顿下山法
            self._newton_downhill_()
        elif self.method == "multiroot":  # 重根情形
            self._multiple_root_()
        else:
            raise ValueError("仅支持newton, halley, simple, downhill, multiroot")
        # 便于索引取值操作，转化为ndarray格式
        self.root_precision_info = np.asarray(self.root_precision_info)
        self.root = self.root_precision_info[-1, 1]  # 满足精度的根
        self._display_csv_info()  # 显示信息或存储外部文件
        return self.root

    def _simple_newton_(self):
        """
        简单牛顿法
        :return:
        """
        lambda_ = 1 / self.dfx(self.x0)  # 简化牛顿法的常数
        iter_, sol_tol, x_b, x_n = 0, np.abs(self.fx(self.x0)), self.x0, self.x0
        while sol_tol > self.eps and iter_ < self.max_iter:
            x_n = x_b - lambda_ * self.fx(x_b)  # 简单牛顿迭代法公式
            iter_, sol_tol = iter_ + 1, np.abs(self.fx(x_n))  # 迭代次数加一，更新精度
            x_b = x_n  # 近似根的迭代
            self.root_precision_info.append([iter_, x_n, sol_tol])  # 迭代过程信息存储

    def _newton_(self):
        """
        经典的牛顿法
        :return:
        """
        iter_, sol_tol = 0, np.abs(self.fx(self.x0))  # 初始变量
        x_b, x_n = self.x0, self.x0  # x_b表示x_k, x_n表示x_{k+1}
        while sol_tol > self.eps and iter_ < self.max_iter:
            x_n = x_b - self.fx(x_b) / self.dfx(x_b)  # 牛顿迭代法公式
            iter_, sol_tol = iter_ + 1, np.abs(self.fx(x_n))  # 更新变量
            x_b = x_n  # 近似根的迭代
            # 迭代过程信息存储，格式为[[k, x_k, |f(x_k)|],...]
            self.root_precision_info.append([iter_, x_n, sol_tol])

    def _newton_halley_(self):
        """
        牛顿加速哈利法
        :return:
        """
        iter_, sol_tol = 0, np.abs(self.fx(self.x0))  # 初始变量
        x_b, x_n = self.x0, self.x0  # x_b表示x_k, x_n表示x_{k+1}
        while sol_tol > self.eps and iter_ < self.max_iter:
            f_b, df_b, df2_b = self.fx(x_b), self.dfx(x_b), self.d2fx(x_b)
            # 哈利迭代法公式
            x_n = x_b - f_b / df_b / (1 - f_b * df2_b / (2 * df_b ** 2))
            iter_, sol_tol = iter_ + 1, np.abs(self.fx(x_n))  # 更新变量
            x_b = x_n  # 近似根的迭代
            self.root_precision_info.append([iter_, x_n, sol_tol])

    def _newton_downhill_(self):
        """
        牛顿下山法, 包含有下山因子
        :return:
        """
        iter_, sol_tol = 0, np.abs(self.fx(self.x0))
        x_b, x_n = self.x0, self.x0  # x_b表示x_k, x_n表示x_{k+1}
        downhill_lambda = []  # 存储下山因子
        while sol_tol > self.eps and iter_ < self.max_iter:
            iter_ += 1  # 迭代次数加一
            lambda_, df, df1 = 1, self.fx(x_b), self.dfx(x_b)
            x_n = x_b - df / df1  # 牛顿迭代公式
            sol_tol = np.abs(self.fx(x_n))  # 当前精度
            while sol_tol > np.abs(df):  # 保证下降
                lambda_ /= 2  # 下山因子逐次减半
                x_n = x_b - lambda_ * df / df1  # 牛顿下山法迭代公式
                sol_tol = np.abs(self.fx(x_n))  # 更新精度
            if lambda_ < 1:
                downhill_lambda.append([iter_, lambda_])  # 只存储小于1的下山因子
            x_b = x_n  # 近似根的迭代
            self.root_precision_info.append([iter_, x_n, sol_tol])
        if downhill_lambda:  # 下山因子，仅输出不为1的下山因子
            print("迭代次数及下山因子为：")
            for lambda_ in downhill_lambda:
                print(lambda_[0], ": ", lambda_[1])  # 格式为：k:λ

    def _multiple_root_(self):
        """
        牛顿法重根情形
        :return:
        """
        iter_, sol_tol = 0, np.abs(self.fx(self.x0))  # 初始变量
        x_b, x_n = self.x0, self.x0 # x_b表示x_k, x_n表示x_{k+1}
        while sol_tol > self.eps and iter_ < self.max_iter:
            df, d1f, d2f = self.fx(x_b), self.dfx(x_b), self.d2fx(x_b)
            x_n = x_b - df * d1f / (d1f ** 2 - df * d2f)  # 重根情形牛顿公式
            iter_, sol_tol = iter_ + 1, np.abs(self.fx(x_n))  # 更新变量
            x_b = x_n  # 近似解的迭代
            self.root_precision_info.append([iter_, x_n, sol_tol])

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
            print(self.method, "Iter：%d, x = %.20f, Precision：%.15e" % (info[0], info[1], info[2]))
