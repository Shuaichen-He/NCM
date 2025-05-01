# -*- coding: UTF-8 -*-
"""
@file_name: exp1.py
@time: 2023-02-09
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import matplotlib.pyplot as plt
from interpolation_02.lagrange_interpolation import LagrangeInterpolation
from interpolation_02.newton_diff_quotient_interp import NewtonDifferenceQuotient
from interpolation_02.newton_difference_interpolation import NewtonDifferenceInterpolation

x = np.array([460, 760, 1060, 1360, 1660])  # 插值节点
y = np.array([7.04, 4.28, 3.40, 2.54, 2.13])  # 插值节点
x0 = np.array([500, 600, 1000, 1450])  # 待求插值点

# 1. 拉格朗日插值
lagI = LagrangeInterpolation(x, y)  # 初始化对象
lagI.fit_interp()  # 执行插值
y0 = lagI.predict_x0(x0)  # 推测未知数据的插值
print("拉格朗日插值多项式：\n", lagI.polynomial)  # 打印拉格朗日多项式特征
print("多项式系数：", lagI.poly_coefficient)  # 系数
print("多项式系数对应的各阶次：", lagI.coefficient_order)   # 对应阶次
print("插值点x0的多项式插值：", y0)    # 推测的插值
print("=" * 80)

# 2. 牛顿差商插值
NDQ = NewtonDifferenceQuotient(x, y)
NDQ.fit_interp()
y0 = NDQ.predict_x0(x0)
print("牛顿差商插值多项式：\n", NDQ.polynomial)  # 打印牛顿差商插值多项式特征
print("多项式系数：", NDQ.poly_coefficient)  # 系数
print("多项式系数对应的各阶次：", NDQ.coefficient_order)  # 对应阶次
print("插值点x0的多项式插值：", y0)  # 推测的插值
print("=" * 80)

# 3. 牛顿差分插值
NDI_f = NewtonDifferenceInterpolation(x, y, diff_method="forward")
NDI_f.fit_interp()
y0 = NDI_f.predict_x0(x0)
print("牛顿前向差分插值多项式：\n", NDI_f.polynomial)  # 打印牛顿前向差分插值多项式特征
print("多项式系数：", NDI_f.poly_coefficient)  # 系数
print("多项式系数对应的各阶次：", NDI_f.coefficient_order)  # 对应阶次
print("插值点x0的多项式插值：", y0)  # 推测的插值
print("=" * 80)

NDI_b = NewtonDifferenceInterpolation(x, y, diff_method="backward")
NDI_b.fit_interp()
y0 = NDI_b.predict_x0(x0)
print("牛顿后向差分插值多项式：\n", NDI_b.polynomial)  # 打印牛顿后向差分插值多项式特征
print("多项式系数：", NDI_b.poly_coefficient)  # 系数
print("多项式系数对应的各阶次：", NDI_b.coefficient_order)  # 对应阶次
print("插值点x0的多项式插值：", y0)  # 推测的插值
print("=" * 80)

# 可视化
plt.figure(figsize=(14, 10))
plt.subplot(221)
lagI.plt_interpolation(x0, y0, is_show=False)
plt.subplot(222)
NDQ.plt_interpolation(x0, y0, is_show=False)
plt.subplot(223)
NDI_f.plt_interpolation(x0, y0, is_show=False)
plt.subplot(224)
NDI_b.plt_interpolation(x0, y0, is_show=False)
plt.show()