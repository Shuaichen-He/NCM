# -*- coding: UTF-8 -*-
"""
@file_name: sqrt_iteration_op.py
@time: 2022-10-31
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import math  # 基本数值运算库，标量值的计算
import warnings  # 警告信息库
import matplotlib.pyplot as plt  # 导入 pyplot 模块，主要用于绘图，以及修饰图形，别名 plt
import matplotlib as mpl  # 导入 matplotlib 绘图库，别名 mpl

# 数学公式字体，支持['dejavusans','dejavuserif','cm','stix','stixsans','custom']
plt.rcParams["mathtext.fontset"] = "cm"  # # 设置数学模式下的字体格式
# 中文常见字体格式：SimHei黑体，Kaiti楷体，LiSu隶书，FansSong仿宋，YouYuan幼圆，STSong华文宋体
# 中文字体：支持['SimHei', 'Kaiti', 'LiSu', 'FansSong', 'YouYuan', 'STSong']
mpl.rcParams["font.family"] = "STSong"  # 中文显示，此处仿宋
plt.rcParams["axes.unicode_minus"] = False  # 解决坐标轴负数的负号显示问题


def check_params_condition(a, x0):
    """
    参数条件的判断：要求开方数a为数值正数，迭代初值x0为正数
    :param a: 开方数
    :param x0: 迭代初始值
    :return:
    """
    # 如下健壮性判断为选择结构，要求开方数必须为数值且是正数
    if type(a) not in [float, int] or a < 0.0:  # a应为数值且是正数
        raise ValueError("开方数应为数值且是正数.")  # 人为引发一个异常值，终止执行
    # 如下健壮性判断为选择结构，要求迭代初始值不能为0，且尽可能是正数（放宽了条件）
    if x0 == 0.0:  # 迭代初始值，不能为零，因为要作为分母
        raise ValueError("迭代初始值不能为零.")
    elif x0 < 0.0:  # 如果为负数，则取绝对值，然后提示警告信息
        # 警告信息，不终止执行，需导入import warnings
        warnings.warn("迭代初始值x0不能为负数，此处按绝对值|x0|逼近.")
        return abs(x0)  # 取绝对值
    else:
        return x0


def sqrt_cal_while(a, x0, eps=1e-15, max_iter=100):
    """
    核心算法：迭代逼近，while循环结构。迭代公式为：x_(k+1) = 0.5 * (x_k + a / x_k)
    :param a: 开方数，数值正数
    :param x0: 迭代初始值， 大于0
    :param eps:  # 开方运算的终止精度
    :param max_iter: 开方运算的最大迭代次数
    :return: 最终满足精度的近似值x_k，以及迭代过程中的近似值approximate_values
    """
    x0 = check_params_condition(a, x0)  # 输入参数的判断
    approximate_values = [x0]  # 迭代逼近过程中的值
    x_k, tol, iter_ = x0, math.inf, 0  # 初始化，迭代值x_k、逼近精度tol和迭代次数iter_
    # 如下采用while循环结构，若满足任何一个条件（精度要求和最大迭代次数），则继续迭代
    while tol >= eps and iter_ < max_iter:
        x_b = x_k  # x_b为迭代的上一次值
        x_k = (x_k + a / x_k) / 2  # 开方运算迭代公式
        approximate_values.append(x_k)  # 存储迭代过程中的近似值
        tol = abs(x_k - x_b)  # 相邻两次迭代的绝对差值为精度，改变量较小时，终止
        iter_ += 1  # 迭代次数加一
    return x_k, approximate_values


def sqrt_cal_for(a, x0, eps=1e-15, max_iter=100):
    """
    核心算法：迭代逼近，for循环结构。迭代公式为：x_(k+1) = 0.5 * (x_k + a / x_k)
    :return: 最终满足精度的近似值x_k，以及迭代过程中的近似值approximate_values
    """
    x0 = check_params_condition(a, x0)  # 输入参数的判断
    approximate_values = [x0]  # 迭代逼近过程中的值
    x_k, tol, iter_ = x0, math.inf, 0  # 初始化，迭代值x_k、逼近精度tol和迭代次数iter_
    # 如下采用for循环结构，在最大迭代次数内逐次逼近，每次计算精度，若满足，则终止循环（迭代）
    for _ in range(max_iter):  # 无需循环变量，故用“_”忽略
        x_b = x_k  # x_b为迭代的上一次值
        x_k = (x_k + a / x_k) / 2  # 开方运算迭代公式
        approximate_values.append(x_k)  # 存储迭代过程中的近似值
        if abs(x_k - x_b) <= eps:  # 相邻两次迭代的绝对差值为精度，改变量较小时，终止
            break
    return x_k, approximate_values


def plt_approximate_processing(appr_values, is_show=True):
    """
    可视化开方迭代逼近过程中的近似值曲线
    :param appr_values: 迭代逼近过程中的近似值列表
    :param is_show: 是否可视化，用于绘制子图，子图时值设置为False
    :return:
    """
    if is_show:
        plt.figure(figsize=(7, 5))
    plt.plot(appr_values, "ko--", label="$x_k, k=%d$" % (len(appr_values) - 1))  # 可视化开方近似值
    plt.plot(appr_values[0], "D",
             label="$x_0=%.f, \ \epsilon=10^{-16}$" % appr_values[0])  # 可视化初值
    plt.plot(len(appr_values) - 1, appr_values[-1], "s",
             label="$x^* = %.15f$" % appr_values[-1])  # 可视化最终近似值
    plt.xlabel("Iterations$(k)$", fontdict={"fontsize": 18})  # x轴标记
    plt.ylabel(r"$x_k(\approx \sqrt{a})$", fontdict={"fontsize": 18})  # y轴标记
    plt.text(2, 14, r"迭代公式：", fontdict={"fontsize": 18})  # 无指向型注释text()
    # 无指向型注释text(), 注意latex修饰
    plt.text(2, 12, r"$x_{k+1} = \dfrac{1}{2}\left ( x_k + \dfrac{a}{x_k} \right),"
                    r"\ k = 0,1,2,\cdots$", fontdict={"fontsize": 18})
    tol_ = abs(appr_values[-1] - appr_values[-2])  # 算法终止的精度
    plt.title(r"开方运算：$\epsilon=\vert x_{k+1} - x_{k} \vert = %.5e$" % tol_,
              fontdict={"fontsize": 18})  # 标题
    plt.legend(frameon=False, fontsize=16, loc="best")  # 添加图例
    plt.tick_params(labelsize=16)  # 刻度字体大小16
    plt.grid(ls=":")  # 添加网格线，且是虚线
    if is_show:
        plt.show()


def plt_approximate_processing2(appr_values, is_show=True):
    """
    可视化开方迭代逼近过程中的近似值曲线
    :param appr_values: 迭代逼近过程中的近似值列表
    :param is_show: 是否可视化，用于绘制子图，子图时值设置为False
    :return:
    """
    if is_show:
        plt.figure(figsize=(7, 5))
    plt.plot(appr_values, "ko:", lw=2, label="$x_k, k=%d$" % (len(appr_values) - 1))  # 可视化开方近似值
    plt.plot(appr_values[0], "D", label="$x_0=%.f, \ \epsilon=10^{-16}$" % appr_values[0])  # 可视化初值
    plt.plot(len(appr_values) - 1, appr_values[-1], "s", label="$x^* = %.15f$" % appr_values[-1])
    plt.xlabel("Iterations$(k)$", fontdict={"fontsize": 18})  # x轴标记
    plt.ylabel(r"$x_k(\approx \sqrt{a})$", fontdict={"fontsize": 18})  # y轴标记
    plt.text(3, 12, r"迭代公式：", fontdict={"fontsize": 18})
    plt.text(3, 10, r"$x_{k+1} = \dfrac{1}{2}\left ( x_k + \dfrac{a}{x_k} \right),\ k = 0,1,2,\cdots$",
             fontdict={"fontsize": 18})
    tol_ = abs(appr_values[-1] - appr_values[-2])  # 算法终止的精度
    plt.title(r"开方运算：$\epsilon=\vert x_{k+1} - x_{k} \vert = %.5e$" % tol_, fontdict={"fontsize": 18})
    plt.legend(frameon=True, fontsize=16, loc="best")  # 添加图例
    plt.tick_params(labelsize=16)  # 刻度字体大小16
    plt.xticks([k for k in range(len(appr_values))])
    # plt.grid(ls=":")
    if is_show:
        plt.show()
