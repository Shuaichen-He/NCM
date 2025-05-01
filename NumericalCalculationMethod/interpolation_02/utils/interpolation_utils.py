# -*- coding: UTF-8 -*-
"""
@file_name: interpolation_entity_utils.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np  # 数值计算
import sympy  # 符号计算
from util_font import *


class InterpolationUtils:
    """
    多项式插值工具类，封装插值多项式的类属性以及常见工具实例方法
    """
    # 类属性变量：
    polynomial, poly_degree = None, None  # 插值多项式和多项式系数最高阶
    poly_coefficient, coefficient_order = None, None  # 多项式系数和各系数阶次

    def __init__(self, x, y):
        """
        多项式插值必要参数初始化，及各健壮性条件测试
        :param x: 已知自变量数据，格式可为列表、ndarray
        :param y: 已知因变量数据，格式可为列表、ndarray
        """
        self.x = np.asarray(x, dtype=np.float64)  # 显式转换为ndarray
        self.y = np.asarray(y, dtype=np.float64)  # 显式转换为ndarray
        if len(self.x) < 2 or len(self.x) != len(self.y):
            raise ValueError("数据(xi,yi)的维度不匹配或插值节点数量过少。")
        self.n = len(x)  # 已知数据节点的数量

    def interpolation_polynomial(self, t):
        """
        插值多项式的特征项
        :return:
        """
        if self.polynomial:
            self.polynomial = sympy.expand(self.polynomial)  # 多项式展开
            polynomial = sympy.Poly(self.polynomial, t)  # 生成多项式对象
            self.poly_coefficient = polynomial.coeffs()  # 获取多项式的系数
            self.poly_degree = polynomial.degree()  # 获得多项式的最高阶次
            self.coefficient_order = polynomial.monoms()  # 多项式的阶次
        else:
            print("插值多项式的类属性polynomial为None.")
            exit(0)

    def predict_x0(self, x0):
        """
        预测，通过离散插值点生成的插值多项式（符号多项式），计算插值点x0的插值
        :param x0: 所求插值点，结构可为元组、列表或ndarray对象
        :return:
        """
        x0 = np.asarray(x0, dtype=np.float64)  # 显示转化为ndarray
        if self.polynomial:
            t = self.polynomial.free_symbols.pop()  # 获取插值多项式的自由符号变量
            # 转换为lambda函数，并进行数值计算
            lambda_f = sympy.lambdify(t, self.polynomial, "numpy")
            return lambda_f(x0)
        else:
            return None

    def plt_interpolation(self, params, fh=None):
        """
        可视化插值多项式，以及插值点
        :param params: 可视化必要参数信息元组
        :param fh: 模拟函数
        :return:
        """
        title_info, x0, y0, is_show = params  # 解包
        if is_show:  # 用于子图绘制，如果当前图形绘制为一子图，则is_show设置为False
            plt.figure(figsize=(7, 5))

        xi = np.linspace(min(self.x), max(self.x), 200)  # 插值区间内等分150个离散插值节点
        yi_hat = self.predict_x0(xi)  # 求等分点的插值
        plt.plot(xi, yi_hat, "k-", label="$g(x)$曲线")  # 可视化插值多项式
        if x0 is not None and y0 is not None:
            plt.plot(x0, y0, "bs", markersize=6, label="$(x_0, \hat y_0)$")  # 可视化所求插值点
        mse = 0.0  # 均方误差
        if fh is not None:
            plt.plot(xi, fh(xi), "r--", label="$f(x)$曲线")  # 真实函数曲线
            mse = np.mean((fh(xi) - yi_hat) ** 2)  # 均方误差
        plt.plot(self.x, self.y, "ro", label="$(x_i,y_i)$")  # 离散插值节点
        plt.legend(frameon=False, fontsize=16)  # 添加图例，并取消外方框
        plt.grid(ls=":")  # 添加主要网格线，且是虚线
        plt.xlabel("$x$", fontdict={"fontsize": 18})  # 横坐标标记，latex修饰
        plt.ylabel("$f(x) \ /\  g(x)$", fontdict={"fontsize": 18})  # 纵坐标标记，latex修饰
        plt.tick_params(labelsize=16)  # 刻度字体大小16
        if mse != 0.0:
            plt.title(title_info + "插值多项式：$MSE=%.5e$" % mse, fontdict={"fontsize": 18})  # 标题
        else:
            plt.title(title_info + "插值多项式曲线及插值节点", fontdict={"fontsize": 18})  # 标题
        if is_show:
            plt.show()

    def check_equidistant(self):
        """
        判断数据节点x是否是等距节点. 若等距, 返回等距步长h
        :return:
        """
        if self.n < 2:
            raise ValueError("插值节点数量最少为2个。。。")
        xx = np.linspace(min(self.x), max(self.x), self.n, endpoint=True)
        if (self.x == xx).all() or (self.x == xx[::-1]).all():  # 升序或降序
            return self.x[1] - self.x[0]  # 等距步长
        else:
            raise ValueError("非等距节点，不可使用此算法。")

    def cal_difference(self, diff_method="forward"):
        """
        计算牛顿差分：向前差分forward，向后差分backward
        :return:
        """
        self.check_equidistant()  # 首先判断是否等距节点
        diff_val = np.zeros((self.n, self.n))  # 差分表
        diff_val[:, 0] = self.y  # 第1列存储离散数据值
        if diff_method == "forward":  # 前向差分
            for j in range(1, self.n):
                i = np.arange(0, self.n - j)
                diff_val[i, j] = diff_val[i + 1, j - 1] - diff_val[i, j - 1]
        elif diff_method == "backward":  # 后向差分
            for j in range(1, self.n):
                i = np.arange(j, self.n)
                diff_val[i, j] = diff_val[i, j - 1] - diff_val[i - 1, j - 1]
        else:
            raise AttributeError("仅支持forward、backward两种差分.")
        return diff_val
