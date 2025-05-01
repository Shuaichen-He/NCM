# -*- coding: UTF-8 -*-
"""
@file: test_lagrange_poly.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from interpolation_02.lagrange_interpolation import LagrangeInterpolation
from util_font import *


def plt_lagrange(x, y, x0, y0, lag_obj, fh, info):
    """
    可视化函数
    :return:
    """
    plt.plot(x, y, "ro", label="$(x_i,y_i)$")  # 离散插值节点
    xi = np.linspace(min(x), max(x), 200, endpoint=True)  # 插值区间[a, b]内等分100个离散插值节点
    yi_hat = lag_obj.predict_x0(xi)  # 求等分点的插值
    yi = fh(xi)  # 真值
    plt.plot(xi, yi_hat, "r-", label="$%s$曲线" % info[0])  # 可视化插值多项式
    plt.plot(xi, yi, "k--", label="$%s$曲线" % info[1])  # 可视化模拟函数曲线
    plt.plot(x0, y0, "bs", markersize=6, label="$(x_0, \hat y_0)$")  # 可视化所求插值点
    plt.legend(frameon=False, fontsize=16, loc="best")  # 添加图例，并取消外方框
    plt.grid(ls=":")  # 添加主要网格线，且是虚线
    plt.xlabel("$x$", fontdict={"fontsize": 18})  # 横坐标标记，latex修饰
    plt.ylabel("$f(x) \ /\  g(x)$", fontdict={"fontsize": 18})  # 纵坐标标记，latex修饰
    plt.tick_params(labelsize=16)  # 刻度字体大小16
    mse = np.mean((yi - yi_hat) ** 2)  # 均方误差
    plt.title("$Lagrange$插值：$MSE=%.5e$" % mse, fontdict={"fontsize": 18})  # 标题


fh1 = lambda x: 0.5 * x ** 5 - x ** 4 + 1.8 * x ** 3 - 3.6 * x ** 2 + 4.5 * x + 2  # 模拟函数1
fh2 = lambda x: 11 * np.sin(x) + 7 * np.cos(5 * x)  # 模拟函数2

x = np.linspace(-1, 3, 10, endpoint=True)
y = fh1(x)  # 取值模拟
x0 = np.asarray([-0.9, -0.2, 1.5, 2.2, 2.7, 2.9])
lag = LagrangeInterpolation(x, y)
lag.fit_interp()
y0 = lag.predict_x0(x0)  # 预测
# 打印拉格朗日多项式特征
print("拉格朗日插值多项式：\n", lag.polynomial)
print("多项式系数：", lag.poly_coefficient)
print("多项式系数对应的各阶次：", lag.coefficient_order)
for i in range(len(y0)):
    print("%.6f" % y0[i])
print("插值点x0的多项式插值：", y0, "，误差：", fh1(x0) - y0)
print("离散数据点(xk, yk)对应的多项式插值\n", lag.predict_x0(x))
print("离散数据点x插值的误差：\n", fh1(x) - lag.predict_x0(x))
print(lag.interp_base_fun)
print("=" * 80)

plt.figure(figsize=(14, 5))
plt.subplot(121)
plt_lagrange(x, y, x0, y0, lag, fh1, ["g_1(x)", "f_1(x)"])
plt.subplot(122)
y = fh2(x)
lag2 = LagrangeInterpolation(x, y)
lag2.fit_interp()
y0 = lag2.predict_x0(x0)  # 预测
plt_lagrange(x, y, x0, y0, lag2, fh2, ["g_2(x)", "f_2(x)"])
plt.show()

print("多项式系数：", lag2.poly_coefficient)
print("多项式系数对应的各阶次：", lag2.coefficient_order)
print("插值点x0的多项式插值：", y0, "，误差：", fh2(x0) - y0)
print("离散数据点(xk, yk)对应的多项式插值\n", lag2.predict_x0(x))
print("离散数据点x插值的误差：\n", fh2(x) - lag2.predict_x0(x))
print(lag.interp_base_fun)
print("=" * 80)