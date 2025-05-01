# -*- coding: UTF-8 -*-
"""
@file:interval_segmentation_method.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import pandas as pd
import datetime


class IntervalSegmentation_Root:
    """
    区间分割发求解：二分法，试值法两种方法。
    自适应划分，满足精度要求即可，未设置最大迭代次数以及划分区间数上限变量。
    """

    def __init__(self, fx, x_span, eps=1e-15, display="display", funEval="dichotomy"):
        self.fx = fx  # 待求根方程
        self.a, self.b = x_span[0], x_span[1]  # 求解区间
        self.eps = eps  # 近似根的精度要求
        self.display = display  # 值有to_csv（存储外部文件），display（只显示最终结果）
        self.method = funEval  # 求解方法，默认为二分法
        self.root_precision_info = []  # 存储划分的区间，近似根，精度
        self.root = None  # 最终近似根

    def _solve_root(self, a, b):
        """
        把区间端点带入方程，区间端点a和b不断更新，区间[a, b]不断缩小
        :return:
        """
        fa_val, fb_val = self.fx(a), self.fx(b)  # 左右端点函数值
        if fa_val * fb_val > 0:
            raise ValueError("两端点函数值乘积大于0，不存在根！")
        fm_val = self.fx((a + b) / 2)  # 区间中点函数值
        self.root_precision_info.append([a, b, (a + b) / 2, fm_val])  # 构建存储区间划分过程
        return fa_val, fb_val, fm_val

    def fit_root(self):
        """
        区间分割法非线性方程求根
        :return:
        """
        a, b = self.a, self.b
        if self.method.lower() == "regula":
            self._regula_falsi_method_(a, b)  # 试值法或试位法
        else:
            fa_val, fb_val, fm_val = self._solve_root(a, b)  # 获取两端点和中点的函数值
            if abs(fa_val) <= self.eps:  # 左端点函数值满足精度要求
                self.root = fa_val  # 左端点即为近似根
                return  # 直接返回，无需再分割区间
            elif abs(fb_val) <= self.eps:  # 右端点函数值满足精度要求
                self.root = fb_val  # 右端点即为近似根
                return  # 直接返回，无需再分割区间
            # 二分法在精度要求下循环划分区间
            while abs(fm_val) > self.eps and abs(b - a) > self.eps:  # 此处加入解的真实精度判别
                if fa_val * fm_val < 0:
                    b = (a + b) / 2  # 取前半区间
                else:
                    a = (a + b) / 2  # 取后半区间
                # 在新的区间端点求方程值
                fa_val, fb_val, fm_val = self._solve_root(a, b)
        self.root_precision_info = np.asarray(self.root_precision_info)  # 便于索引取值操作
        self.root = self.root_precision_info[-1, 2]  # 满足精度的根
        self._display_csv_info()  # 显示信息或存储外部文件
        return self.root

    def _display_csv_info(self):
        """
        求解过程的显示控制，以及把迭代信息存储到外部文件
        :return:
        """
        if self.display.lower() == "to_csv":
            res = pd.DataFrame(self.root_precision_info,
                               columns=["left", "right", "root", "precision"])
            res.to_csv("../result_file/result%s.csv" %
                       datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
        elif self.display.lower() == "display":  # 显示
            final_info = self.root_precision_info[-1, :]  # 最终的信息
            print("最终分割区间：[%.10f, %.10f], x = %.20f, 精度：%.10e, 迭代次数: %d"
                  % (final_info[0], final_info[1], self.root, final_info[-1],
                     len(self.root_precision_info)))

    def _regula_falsi_method_(self, a, b):
        """
        试值法或试位法求解
        :return:
        """
        fa_val, fb_val = self.fx(a), self.fx(b)  # 区间端点的函数值
        lx_point = b - (fb_val * (b - a)) / (fb_val - fa_val)  # 割线L与x轴交点
        fc_eps = self.fx(lx_point)  # lx_point点的方程值误差
        self.root_precision_info.append([a, b, lx_point, fc_eps])  # 构建存储区间划分过程
        if abs(fc_eps) <= self.eps:
            self.root = lx_point
        while abs(fc_eps) > self.eps:  # 纵坐标判别准则
            if fb_val * fc_eps > 0:
                b, fb_val = lx_point, fc_eps  # 在[a, lx_point]内有一个零点
            else:
                a, fa_val = lx_point, fc_eps  # 在[lx_point, b]内有一个零点
            lx_point = b - (fb_val * (b - a)) / (fb_val - fa_val)  # 割线L与x轴交点
            fc_eps = self.fx(lx_point)  # c点的方程值误差
            self.root_precision_info.append([a, b, lx_point, fc_eps])  # 构建存储区间划分过程
            if np.min([abs(lx_point), lx_point - a]) <= self.eps:  # 横坐标判别准则
                break
