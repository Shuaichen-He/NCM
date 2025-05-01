# -*- coding: UTF-8 -*-
"""
@file:runge_phenomenon.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from util_font import *
from interpolation_02.lagrange_interpolation import LagrangeInterpolation
import sympy


runge_fun = lambda x: 1 / (x ** 2 + 1)  # 定义龙格函数

plt.figure(figsize=(14, 5))
plt.subplot(121)
x0 = np.linspace(-5, 5, 200, endpoint=True)  # 模拟绘制图像的插值节点
lw_ = [1, 1.2, 1.3, 1.5, 1.7]  # 线条宽度
ls_ = ["-", "--", ":", "-.", "-"]  # 线性类型
for i, n in enumerate(range(3, 12, 2)):
    x_k = np.linspace(-5, 5, n, endpoint=True)  # [-5, 5]区间等分数据，以获得离散插值点xk
    y_k = runge_fun(x_k)  # 在等分点xk处的函数值
    lag_interp = LagrangeInterpolation(x_k, y_k)  # 构造拉格朗日插值对象
    lag_interp.fit_interp()  # 生成拉格朗日插值多项式
    y0 = lag_interp.predict_x0(x0)  # 插值节点的拉格朗日插值多项式值
    plt.plot(x0, y0, ls_[i],  lw=lw_[i], label=r"$n = %d$" % (n - 1))  # 绘制图像
    print(lag_interp.poly_coefficient)

plt.plot(x0, runge_fun(x0), "k-", lw=1.9, label="$Runge \ Function$")  # 龙格函数曲线
plt.xlabel(r"$x$", fontdict={"fontsize": 18})
plt.ylabel(r"$g(x) \ / \ f(x)$", fontdict={"fontsize": 18})
plt.tick_params(labelsize=16)  # 刻度字体大小14
plt.title("龙格函数在不同阶次下的插值多项式曲线", fontdict={"fontsize": 18})
plt.legend(frameon=False, fontsize=16, loc="best", ncol=2)

plt.subplot(122)
mse_list = []
poly20, poly4 = None, None  # 记录阶次为4、20的多项式
for i, n in enumerate(range(3, 22, 1)):
    print(n)
    x_k = np.linspace(-5, 5, n, endpoint=True)  # [-5, 5]区间等分数据，以获得离散插值点xk
    y_k = runge_fun(x_k)  # 在等分点xk处的函数值
    lag_interp = LagrangeInterpolation(x_k, y_k)  # 构造拉格朗日插值对象
    lag_interp.fit_interp()  # 生成拉格朗日插值多项式
    y0 = lag_interp.predict_x0(x0)  # 插值节点的拉格朗日插值多项式值
    mse_list.append(np.mean((runge_fun(x0) - y0) ** 2))  # 均方误差
    if n == 21:
        poly20 = lag_interp.polynomial
    if n == 5:
        poly4 = lag_interp.polynomial
plt.semilogy(np.arange(3, 22, 1) - 1, mse_list, "-", lw=1)  # mse曲线
plt.semilogy(np.arange(3, 22, 1) - 1, mse_list, "o")  # mse曲线
plt.xlabel(r"$Orders(n)$", fontdict={"fontsize": 18})
plt.ylabel(r"$MSE$", fontdict={"fontsize": 18})
plt.tick_params(labelsize=16)  # 刻度字体大小14
plt.title("龙格函数在不同阶次下插值多项式的$MSE$", fontdict={"fontsize": 18})
plt.grid(ls=":")
plt.xticks(np.linspace(2, 20, 10))
plt.show()
print("均方误差：", mse_list)


# 高阶多项式的扰动问题
x0 = np.linspace(-5, 5, 300, endpoint=True)  # 模拟绘制图像的插值节点
mae4_list, mae20_list = [], []
t = sympy.symbols("t")  # 符号变量
poly_4 = sympy.lambdify(t, poly4, "numpy")
poly_20 = sympy.lambdify(t, poly20, "numpy")
y4_true, y20_true = poly_4(x0), poly_20(x0)  # 假设为真值
for i in range(500):
    idx = np.random.randint(0, 299, 10)
    x0_e = np.copy(x0)
    x0_e[idx] = x0[idx] + 0.001 * np.random.randn(10)
    y4_e = poly_4(x0_e)  # 扰动后的值
    y20_e = poly_20(x0_e)  # 扰动后的值
    # mae_4 = np.mean(np.abs((y4_e - y4_true) / y4_true))
    # mae_20 = np.mean(np.abs((y20_e - y20_true) / y20_true))
    mae_4 = np.mean(np.abs((y4_e - y4_true)))
    mae_20 = np.mean(np.abs((y20_e - y20_true)))
    # print(mae_k[-1], mae_20, np.abs(mae_20 - mae_k[-1]) / mae_k[-1])
    mae4_list.append(mae_4)
    mae20_list.append(mae_20)
print("平均：", np.mean(mae4_list), np.mean(mae20_list), "，比值：", np.mean(mae20_list) / np.mean(mae4_list))

plt.figure(figsize=(14, 5))
plt.subplot(121)
plt.semilogy(np.arange(1, 501), mae4_list, ":", lw=2)
plt.semilogy(np.arange(1, 501), np.mean(mae4_list) * np.ones(500), "-", lw=2,
             label="$\mu=%.2e(\pm%.2e)$" % (np.mean(mae4_list), np.std(mae4_list)))
plt.xlabel("试验次数$k$", fontsize=18)
plt.ylabel("$MAE_k$", fontsize=18)
plt.legend(frameon=False, fontsize=18, ncol=2)
plt.tick_params(labelsize=16)
plt.title("对$x$的微小扰动$g_{4}(x)$的扰动变化", fontsize=18)
plt.ylim([1e-6, 1e-2])
plt.subplot(122)
plt.semilogy(np.arange(1, 501), mae20_list, ":", lw=2)
plt.semilogy(np.arange(1, 501), np.mean(mae20_list) * np.ones(500), "-", lw=2,
             label="$\mu=%.2e(\pm%.2e)$" % (np.mean(mae20_list), np.std(mae20_list)))
plt.xlabel("试验次数$k$", fontsize=18)
plt.ylabel("$MAE_k$", fontsize=18)
plt.legend(frameon=False, fontsize=18, ncol=2)
plt.tick_params(labelsize=16)
plt.title("对$x$的微小扰动$g_{20}(x)$的扰动变化", fontsize=18)
plt.ylim([1e-6, 1e-2])
plt.show()