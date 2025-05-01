# -*- coding: UTF-8 -*-
"""
@file_name: piecewise_interp_utils.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from util_font import *


class PiecewiseInterpUtils:
    """
    分段插值实体类，封装插值多项式的类属性以及常见工具实例方法
    """
    # 类属性变量：
    polynomial = None  # 分段插值多项式，字典形式，即每区间一个多项式
    poly_coefficient = None  # 多项式系数矩阵

    def __init__(self, x, y):
        """
        多项式插值必要参数初始化，及各健壮性条件测试
        :param x: 已知离散数据的x坐标点
        :param y: 已知离散数据的y坐标点
        """
        self.x = np.asarray(x, dtype=np.float64)  # 显式转换ndarray，方便后续数值运算
        self.y = np.asarray(y, dtype=np.float64)
        if len(self.x) > 1 and len(self.x) == len(self.y):
            self.n = len(x)  # 已知数据节点的数量
        else:
            raise ValueError("数据(xi,yi)的维度不匹配或插值点数量过少.")

    def predict_x0(self, x0):
        """
        预测，通过离散插值点生成的分段插值多项式（符号多项式），计算插值点x0的插值
        :param x0: 所求插值点，格式可为元组、列表或ndarray对象
        :return:
        """
        if self.polynomial:
            x0 = np.asarray(x0, dtype=np.float64)  # 类型转换
            y_0 = np.zeros(len(x0))  # 存储x0的插值
            t = self.polynomial[0].free_symbols.pop()  # 获取插值多项式的自由符号变量
            # 对每一个插值点x0求解插值，首先查找所在区间段，然后采用该区间段多项式求解
            idx = 0  # 默认第一个多项式
            for i in range(len(x0)):
                # 查找被插值点x0所处的区间段索引idx
                for j in range(1, self.n - 1):
                    if self.x[j] <= x0[i] <= self.x[j + 1] or \
                            self.x[j] >= x0[i] >= self.x[j + 1]:
                        idx = j  # 当前区间段索引
                        break
                y_0[i] = self.polynomial[idx].evalf(subs={t: x0[i]})  # 计算插值
            return y_0

    def check_equidistant(self):
        """
        判断数据节点x是否是等距节点
        :return:
        """
        if self.n > 1:
            xx = np.linspace(min(self.x), max(self.x), self.n, endpoint=True)
            if (self.x == xx).all() or (self.x == xx[::-1]).all():  # 升序或降序
                return self.x[1] - self.x[0]  # 等距步长
            else:
                raise ValueError("非等距节点，不可使用牛顿差分插值方法。")
        else:
            raise ValueError("插值节点数量最少为2个。。。")

    def plt_interpolation(self, params, fh=None):
        """
        可视化分段插值多项式，以及插值点
        :param params: 可视化必要参数信息元组
        :param fh: 模拟函数
        :return:
        """
        title_info, x0, y0, is_show = params  # 解包
        if is_show:  # 用于子图绘制，如果当前图形绘制为一子图，则is_show设置为False
            plt.figure(figsize=(7, 5))
        plt.plot(self.x, self.y, "ro", label="$(x_i,y_i)$")  # 离散插值节点
        xi = np.linspace(min(self.x), max(self.x), 200)  # 插值区间内等分200个离散插值节点
        yi_hat = self.predict_x0(xi)  # 求等分点的插值
        plt.plot(xi, yi_hat, "k-", label="$g(x)$曲线")  # 可视化插值多项式
        if x0 is not None and y0 is not None:
            plt.plot(x0, y0, "bs", markersize=6, label="$(x_0, \hat y_0)$")  # 可视化所求插值点
        mse = 0.0 # 均方误差
        if fh is not None:
            plt.plot(xi, fh(xi), "r--", label="$f(x)$曲线")  # 真实函数曲线
            mse = np.mean((fh(xi) - yi_hat) ** 2)  # 均方误差
        plt.legend(frameon=False, fontsize=16)  # 添加图例，并取消外方框
        plt.grid(ls=":")  # 添加主要网格线，且是虚线
        plt.xlabel("$x$", fontdict={"fontsize": 18})  # 横坐标标记，latex修饰
        plt.ylabel("$f(x) \ /\  g(x)$", fontdict={"fontsize": 18})  # 纵坐标标记，latex修饰
        # plt.ylabel("$g(x)$", fontdict={"fontsize": 18})  # 纵坐标标记，latex修饰
        # plt.xlabel("$time$", fontdict={"fontsize": 18})  # 横坐标标记，latex修饰
        # plt.ylabel("$temperature$", fontdict={"fontsize": 18})  # 纵坐标标记，latex修饰
        plt.tick_params(labelsize=16)  # 刻度字体大小16
        if mse != 0.0:
            plt.title(title_info + "插值：$MSE=%.5e$" % mse, fontdict={"fontsize": 18}) # 标题
        else:
            plt.title(title_info + "插值曲线及插值节点", fontdict={"fontsize": 18})  # 标题
        if is_show:
            plt.show()

