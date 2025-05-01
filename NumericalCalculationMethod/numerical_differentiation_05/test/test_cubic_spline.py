# -*- coding: UTF-8 -*-
"""
@author:Lenovo
@file:test_cubic_spline.py
@time:2021/08/26
"""
import numpy as np
import matplotlib.pyplot as plt
from numerical_differentiation_05.discrete_data_cubic_spline_differetiation \
    import DiscreteDataCubicSplineDifferential

# fun = lambda x: np.sin(x) * np.exp(-x)
fun = lambda x: np.sin(2 * x) ** 2 * np.exp(-0.5 * x)

np.random.seed(100)
x0 = -3 + np.random.rand(20) * (3 - (-3))   # 非等距
x0 = np.sort(x0)
# x0 = np.linspace(-3, 3, 20)  # 等距

print("待求解微分点：", x0)
ddcsd = DiscreteDataCubicSplineDifferential(x0, fun(x0), boundary_cond="natural")   # 修改边界条件，complete，natural
y0 = ddcsd.predict_diff_x0(x0)
# diff_val = np.exp(-x0) * (np.cos(x0) - np.sin(x0))
dfh = lambda x: -0.5 * np.exp(-0.5 * x) * np.sin(2 * x) ** 2 + 4 * np.exp(-0.5* x) * np.sin(2 * x) * np.cos(2 * x)
print("微分值：", y0, "\n精确值：", dfh(x0), "\n误差：", y0 - dfh(x0))
# dfh = lambda x: np.exp(-x) * (np.cos(x) - np.sin(x))
plt.figure(figsize=(14, 5))
plt.subplot(121)
ddcsd.plt_differentiation([min(x0), max(x0)], dfh, x0, y0, is_show=False)
plt.title("三次样条插值求解离散数据数值微分（$20$）", fontdict={"fontsize": 18})
plt.subplot(122)
# x0 = np.linspace(-3, 3, 40)  # 等距
x0 = -3 + np.random.rand(40) * (3 - (-3))  # 非等距
x0 = np.sort(x0)
ddcsd = DiscreteDataCubicSplineDifferential(x0, fun(x0), boundary_cond="natural")    # 修改边界条件
y0 = ddcsd.predict_diff_x0(x0)
print("=" * 80)
print("微分值：", y0, "\n精确值：", dfh(x0), "\n误差：", y0 - dfh(x0))
ddcsd.plt_differentiation([min(x0), max(x0)], dfh, x0, y0, is_show=False, is_fh_marker=False)
plt.title("三次样条插值求解离散数据数值微分（$40$）", fontdict={"fontsize": 18})
plt.show()