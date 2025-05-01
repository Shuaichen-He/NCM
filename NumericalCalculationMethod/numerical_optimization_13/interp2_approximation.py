# -*- coding: UTF-8 -*-
"""
@file_name: interp2_approximation.py
@time: 2022-09-10
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from util_font import *


class Interp2ApproximationOptimization:
    """
    逼近方法，求解单变量函数的极值问题。2次插值逼近
    """

    def __init__(self, fun, x_span, eps, max_iter=1000, is_minimum=True):
        self.fun = fun  # 优化函数
        self.a, self.b = x_span[0], x_span[1]  # 单峰区间
        self.eps, self.max_iter = eps, max_iter  # 精度要求eps和最大迭代次数max_iter
        self.is_minimum = is_minimum  # 是否是极小值，极大值设置为False
        self.local_extremum = None  # 搜索过程，极值点

    def fit_optimize(self):
        """
        2次插值逼近：共五种情况，用条件cond标记
        :return:
        """
        p0, h, dh = self.a, 1, 1e-5  # 初始p0为区间起点，步长为1, 微分步长dh
        if np.abs(p0) > 1e+4:
            h = np.abs(p0) / 1e+4  # 初始化步长
        iter_, err, delta, cond = 1, 1, 1e-6, 0
        max_class = 50  # 最大分类情况迭代次数
        local_extremum = [[p0, self.fun(p0)]]  # 极值迭代过程
        while err > self.eps and iter_ < self.max_iter and cond != 5:
            iter_ += 1  # 迭代次数加1
            # 1. 根据p0的一阶导数确定初始的猜测值p1, p2，实际上确定h
            dp0 = (self.fun(p0 + dh) - self.fun(p0 - dh)) / (2 * dh)  # 以中心差商近似导数
            if dp0 > 0:  # 一阶导数大于0
                h = - np.abs(h)  # 应该选择负的步长
            p1, p2 = p0 + h, p0 + 2 * h  # 确定其他两个猜测点
            y0, y1, y2 = self.fun(p0), self.fun(p1), self.fun(p2)  # 对应函数值
            # p_min, y_min = p0, y0  # 当前极值点(p0, f(p0))
            # 2. 按分类方法，选择待插值的三个点，不断修正步长和更新三个点
            m_c, cond = 0, 0  # 初始化迭代变量m_c和条件类别cond
            while m_c < max_class and np.abs(h) > delta and cond == 0:
                if y0 < y1:  # 分类情况3，表明h过大，跳过了极值点
                    p2, y2 = p1, y1  # 检测靠近p0的点，p2更新
                    h /= 2  # 步长减半
                    p1, y1 = p0 + h, self.fun(p0 + h)  # 重新计算其中一个点p1
                else:  # 分类情况2，已满足y1 < y0
                    if y2 < y1:
                        p1, y1 = p2, y2  # 需检测更靠右的点，p1更新
                        h *= 2  # 步长加倍
                        p2, y2 = p0 + 2 * h, self.fun(p0 + 2 * h)  # 重新计算其中一个点p2
                    else:  # 分类情况1
                        cond = -1  # 满足条件，退出循环
                m_c += 1  # 不满足条件的情况下，继续分类
                if np.abs(h) > 1e+6 or np.abs(p0) > 1e+6:  # 步长和p0过大
                    cond = 5
            # 3. 对选择的三点进行二次插值，逼近目标函数，并求逼近极值点步长h_min
            if cond == 5:
                p_min, y_min = p1, self.fun(p1)  # 步长过大，极值点为(p1,f(p1))
            else:  # 3.1 根据三点进行二次插值，求h_min
                d = 4 * y1 - 2 * y0 - 2 * y2  # h_min公式分母
                if d < 0:  # p1比较靠近p0
                    h_min = h * (4 * y1 - 3 * y0 - y2) / d  # 按公式求解
                else:  # p1比较靠近p2
                    h_min = h / 3  # h_min为原步长的1/3
                    cond = 4
                p_min, y_min = p0 + h_min, self.fun(p0 + h_min)  # p_min比p0更靠近极值点
                # 3.2 确定下一个h的大小
                h = np.abs(h)
                h0, h1, h2 = np.abs(h_min), np.abs(h_min - h), np.abs(h_min - 2 * h)
                h = np.min([h0, h1, h2])  # 取最小的
                if h == 0:
                    h = h_min
                if h < delta:
                    cond = 1
                if np.abs(h) > 1e+6 or np.abs(p_min) > 1e+6:
                    cond = 5
                # 3.3 检查极值，精度更新，考虑三点与当前极值点的绝对值差
                e0, e1, e2 = np.abs(y0 - y_min), np.abs(y1 - y_min), np.abs(y2 - y_min)
                e_min = np.min([e0, e1, e2])
                if e_min != 0 and e_min < err:
                    err = e_min
                if e0 != 0 and e1 == 0 and e2 == 0:
                    err = 0
                if err < self.eps:
                    cond = 2
                p0 = p_min  # 以当前极值更新p0
            local_extremum.append([p_min, self.fun(p_min)])  # 存储当前迭代的极值点
            if cond == 2 and h < delta:
                cond = 3
            # 如果相邻两次的极值点的绝对值小于给定精度，则终止搜索
            if np.abs(local_extremum[-1][1] - local_extremum[-2][1]) < self.eps:
                break
        if self.is_minimum:
            self.local_extremum = np.asarray(local_extremum)
        else:
            self.local_extremum = np.asarray(local_extremum)
            self.local_extremum[:, 1] = -1 * self.local_extremum[:, 1]
        return self.local_extremum[-1]

    def plt_optimization(self, plt_zone=None):
        """
        可视化优化过程
        :param plt_zone:  可视化的区间
        :return:
        """
        if plt_zone is not None:
            xi = np.linspace(plt_zone[0], plt_zone[1], 150)
        else:
            xi = np.linspace(self.a, self.b, 150)
        plt.figure(figsize=(14, 5))
        plt.subplot(121)
        if self.is_minimum:
            plt.plot(xi, self.fun(xi), "k-", lw=1.5, label="$f(x)$")
        else:
            plt.plot(xi, -1 * self.fun(xi), "k-", lw=1.5, label="$f(x)$")
        plt.plot(self.local_extremum[-1, 0], self.local_extremum[-1, 1], "ro", label="$(x^*, f(x^*))$")
        plt.xlabel("$x$", fontdict={"fontsize": 18})
        plt.ylabel("$f(x)$", fontdict={"fontsize": 18})
        plt.title("函数局部极值点$(%.10f, %.10f)$"
                  % (self.local_extremum[-1, 0], self.local_extremum[-1, 1]), fontdict={"fontsize": 18})
        plt.legend(frameon=False, fontsize=16)
        plt.tick_params(labelsize=16)  # 刻度字体大小16
        plt.grid(ls=":")
        plt.subplot(122)
        plt.plot(np.arange(1, len(self.local_extremum) + 1), self.local_extremum[:, 1], "o--")
        plt.xlabel("搜索次数", fontdict={"fontsize": 18})
        plt.ylabel("$f(x^*)$", fontdict={"fontsize": 18})
        plt.title("函数极值优化过程，二次插值逼近搜索$%d$次" % len(self.local_extremum), fontdict={"fontsize": 18})
        # plt.legend(frameon=False, fontsize=16)
        plt.tick_params(labelsize=16)  # 刻度字体大小16
        plt.grid(ls=":")
        plt.show()
