# -*- coding: UTF-8 -*-
"""
@file:test_comp_simpson_2d_int_11.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import matplotlib.pyplot as plt
from numerical_integration_04.composite_simpson_2d_integration import CompositeSimpsonDoubleIntegration

int_fun = lambda x, y: np.exp(-x ** 2 - y ** 2)

plt.figure(figsize=(14, 5))
plt.subplot(121)
ci = CompositeSimpsonDoubleIntegration(int_fun, [0, 1], [0, 1], eps=1e-15, increment=20)
ci.fit_2d_int()
print("划分区间数：%d，积分近似值：%.15f" % (ci.sub_interval_num, ci.int_value))
print("误差：", 0.557746285351034 - ci.int_value)
ci.plt_precision(is_show=False, exact_int=0.557746285351034)
plt.subplot(122)
ci = CompositeSimpsonDoubleIntegration(int_fun, [0, 1], [0, 1], eps=1e-15, increment=50)
ci.fit_2d_int()
print("划分区间数：%d，积分近似值：%.15f" % (ci.sub_interval_num, ci.int_value))
print("误差：", 0.557746285351034 - ci.int_value)
ci.plt_precision(is_show=False, exact_int=0.557746285351034)
plt.show()