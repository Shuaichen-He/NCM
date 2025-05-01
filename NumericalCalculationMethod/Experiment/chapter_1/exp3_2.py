# -*- coding: UTF-8 -*-
"""
@file_name: exp3_2.py
@time: 2023-02-09
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from Experiment.util_font import *  # 导入字体文件


def cal_trapezoid_int_vectorization(int_fx, a, b, eps=1e-10, max_split_interval_num=1000):
    """
    矢量化计算核心算法：自适应计算划分区间数，对每个小区间计算梯形面积，近似曲边梯形面积
    :param int_fx: 被积函数，非符号定义
    :param a: 积分下限
    :param b: 积分上限
    :param eps: 积分精度，采用划分前后两次精度的绝对值差判断
    :param max_split_interval_num: 最大划分区间数
    :return: 积分近似值int_val_n，划分区间数split_num，自适应过程中的积分近似值approximate_values
    """
    approximate_values = []  # 自适应过程中的积分近似值
    int_val_n = (b - a) / 2 * (int_fx(a) + int_fx(b))  # 梯形面积
    approximate_values.append(int_val_n)
    tol, split_num = np.infty, 1  # 初始化，逼近精度tol和区间划分数
    while tol > eps and split_num < max_split_interval_num:
        int_val_b = int_val_n  # 分别模拟当前划分区间数的积分值和下一次划分区间数的积分值
        split_num *= 2  # 每次增加一倍的划分数量：1、2、4、8、16、...
        h = (b - a) / split_num  # 小区间步长，等分
        x_k = np.linspace(a, b, split_num + 1)  # 区间端点，为n + 1
        f_xk = int_fx(x_k)  # 区间端点的函数值
        int_val_n = h / 2 * np.sum((f_xk[:-1] + f_xk[1:]))  # 积分值，一维向量相加
        approximate_values.append(int_val_n)
        tol = np.abs(int_val_n - int_val_b)
    return int_val_n, split_num, approximate_values


def cal_trapezoid_int_nvectorization(int_fx, a, b, eps=1e-10, max_split_interval_num=1000):
    """
    非矢量化计算核心算法：自适应计算划分区间数，对每个小区间计算梯形面积，近似曲边梯形面积
    参数同cal_trapezoid_int_vectorization函数
    :return:
    """
    approximate_values = []  # 自适应过程中的积分近似值
    int_val_n = (b - a) / 2 * (int_fx(a) + int_fx(b))  # 梯形面积
    approximate_values.append(int_val_n)
    tol, split_num = np.infty, 1  # 初始化，逼近精度tol和区间划分数
    while tol > eps and split_num < max_split_interval_num:
        int_val_b = int_val_n  # 分别模拟当前划分区间数的积分值和下一次划分区间数的积分值
        split_num *= 2  # 每次增加一倍的划分数量：1、2、4、8、16、...
        h = (b - a) / split_num  # 小区间步长，等分，标量计算
        x_k, f_xk = [], []  # 列表用于存储区间端点x_k和对应的函数值f_xk
        for i in range(split_num + 1):
            x_k.append(a + i * h)  # 端点值
            f_xk.append(int_fx(x_k[-1]))  # 区间端点的函数值
        int_val_n = 0.0  # 积分值
        for i in range(split_num):
            int_val_n += h / 2 * (f_xk[i] + f_xk[i + 1])  # 积分值，标量相加
        approximate_values.append(int_val_n)
        tol = np.abs(int_val_n - int_val_b)
    return int_val_n, split_num, approximate_values


def plt_approximate_processing(int_fx, approximate_values):
    """
    可视化随着划分区间次数的增加，积分近似值的逼近过程
    :param int_fx: 被积函数，非符号定义
    :param approximate_values: 自适应过程中的积分近似值
    :return:
    """
    plt.figure(figsize=(14, 5))
    xi = np.linspace(0, 1, 150)
    yi = int_fx(xi)
    plt.subplot(121)
    plt.plot(xi, yi, "k-", lw=1)
    plt.fill_between(xi, yi, color="c", alpha=0.5)
    plt.xlabel(r"$x$", fontdict={"fontsize": 18})
    plt.ylabel(r"$f(x)$", fontdict={"fontsize": 18})
    plt.title("被积函数的积分区域", fontdict={"fontsize": 18})
    plt.tick_params(labelsize=16)  # 刻度字体大小16
    plt.subplot(122)
    tol_ = np.abs(approximate_values[-1] - approximate_values[-2])
    plt.plot(approximate_values, "ko--", markerfacecolor="r", markeredgecolor="r",
             label="$a^* = %.15f$" % approximate_values[-1])
    plt.xlabel("划分区间数", fontdict={"fontsize": 18})
    plt.ylabel("积分近似值", fontdict={"fontsize": 18})
    plt.title("积分近似值的逼近过程：$tol = %.5e$" % tol_, fontdict={"fontsize": 18})
    plt.legend(frameon=False, fontsize=16, loc="best")  # 添加图例
    plt.tick_params(labelsize=16)  # 刻度字体大小16
    ticks, orders = [], np.arange(0, len(approximate_values), 2)
    for i in orders:
        ticks.append("$2^{%d}$" % i)
    plt.xticks(orders, ticks)
    plt.grid(ls=":")
    plt.show()
