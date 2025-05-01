# -*- coding: UTF-8 -*-
"""
@file:test_piecewise_linear.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import matplotlib.pyplot as plt
from interpolation_02.piecewise_linear_interpolation import PiecewiseLinearInterpolation
from interpolation_02.piecewise_cubic_hermite_interpolation import PiecewiseCubicHermiteInterpolation

fh = lambda x: 50 * np.sin(x) / x  # 测试函数
diff_fh = lambda x: 50 * (x * np.cos(x) - np.sin(x)) / x ** 2  # 测试函数一阶导函数

x = np.linspace(-4 * np.pi, 4 * np.pi, 20, endpoint=True)
y = fh(x)  # 模拟生成20个数据
x0 = np.array([-3.48, -1.69, 0.05, 2.66, 4.08, 4.876])  # 求解插值节点
print("精确值", fh(x0))
# 1. 分段线性插值
pli = PiecewiseLinearInterpolation(x, y)
pli.fit_interp()
print("分段线性插值各区间段的线性函数系数：")
print(pli.poly_coefficient)
y0_linear = pli.predict_x0(x0)
print("插值点的值：", y0_linear, "，误差：", fh(x0) - y0_linear)
print("=" * 60)
# 2. 分段2点3次埃尔米特插值
pchi = PiecewiseCubicHermiteInterpolation(x, y, diff_fh(x))
pchi.fit_interp()
print("分段埃尔米特插值各区间段的线性函数系数：")
print(pchi.poly_coefficient)
y0_hermite = pchi.predict_x0(x0)
print("插值点的值：", y0_hermite, "，误差：", fh(x0) - y0_hermite)
# 3. 可视化图像
plt.figure(figsize=(14, 5))
plt.subplot(121)
pli.plt_interpolation(x0, y0_linear, fh=fh, is_show=False)
plt.subplot(122)
pchi.plt_interpolation(x0, y0_hermite, fh=fh, is_show=False)
plt.show()
