# -*- coding: UTF-8 -*-
"""
@file_name: test_bspline_interp_int_10.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from numerical_integration_04.cubic_bspline_interpolation_integration import CubicBSplineInterpolationIntegration
from interpolation_02.b_spline_interpolation import BSplineInterpolation
from util_font import *

time = np.linspace(0, 23, 24)  # 时间
traffic_flow = np.array([20, 19, 25, 27, 30, 35, 40, 97, 155, 92, 60, 62, 60, 65,
                         43, 70, 79, 140, 134, 98, 62, 80, 55, 35])  # 速度数据
bsi = BSplineInterpolation(time, traffic_flow, boundary_cond="natural")
bsi.fit_interp()
plt.figure(figsize=(14, 5))
plt.subplot(121)
plt.plot(time, traffic_flow, "o-", lw=1.5)
plt.xlabel("$Time$(间隔时刻)", fontdict={"fontsize": 18})
plt.ylabel("$Traffic \ flow$(辆/分)", fontdict={"fontsize": 18})
plt.title("24小时内车辆在某时刻1分钟内通过桥梁数据折线", fontsize=18)
plt.xticks(np.linspace(0, 24, 13))
plt.tick_params(labelsize=16)  # 刻度字体大小16
plt.grid(ls=":")
plt.subplot(122)
bsi.plt_interpolation(is_show=False)
plt.legend(frameon=False, fontsize=16, loc=2)
plt.xlabel("$Time$(间隔时刻)", fontdict={"fontsize": 18})
plt.ylabel("$Traffic \ flow$(辆/分)", fontdict={"fontsize": 18})
plt.xticks(np.linspace(0, 24, 13))
plt.title("三次均匀$B$样条车流量插值曲线", fontsize=18)
plt.show()

apii = CubicBSplineInterpolationIntegration(time, traffic_flow)
apii.fit_int()
print(apii.int_value)
print("一天内通过桥梁的车流量为：%d" % np.ceil(apii.int_value * 60))
