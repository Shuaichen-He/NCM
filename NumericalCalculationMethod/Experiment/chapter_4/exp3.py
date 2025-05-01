# -*- coding: UTF-8 -*-
"""
@file_name: exp3.py
@time: 2023-02-10
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from numerical_integration_04.composite_simpson_2d_integration import CompositeSimpsonDoubleIntegration
from numerical_integration_04.gauss_legendre_2d_integration import GaussLegendreDoubleIntegration

# 自适应复合辛普森公式
int_fun = lambda theta, r: r * np.exp(-r ** 2)
I_f = np.pi * (1 - np.exp(-4))  # 精确值

csdi = CompositeSimpsonDoubleIntegration(int_fun, [0, 2 * np.pi], [0, 2], eps=1e-15, increment=20)
csdi.fit_2d_int()
print("划分区间数：%d，积分近似值：%.15f" % (csdi.sub_interval_num, csdi.int_value))
print("绝对值误差：%.15e" % abs(I_f - csdi.int_value))
csdi.plt_precision(is_show=True, exact_int=I_f)

# 高斯—勒让德二重积分
gl2d = GaussLegendreDoubleIntegration(int_fun, [0, 2 * np.pi], [0, 2], zeros_num=15)
gl2d.cal_2d_int()
print("积分近似值：%.15f" % gl2d.int_value)
print("绝对值误差：%.15e" % abs(I_f - gl2d.int_value))