# -*- coding: UTF-8 -*-
"""
@file_name: test_sqrt_iteration.py
@time: 2022-10-31
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from fundamentals_python_mathematics_01.sqrt_iteration_op import *


a = 36
x_k, approximate_values = sqrt_cal_for(a=a, x0=20, eps=1e-16)
print(x_k)
plt.figure(figsize=(14, 5))
plt.subplot(121)
plt_approximate_processing(approximate_values, is_show=False)
plt.subplot(122)
a = np.pi
x_k, approximate_values = sqrt_cal_for(a=a, x0=20, eps=1e-16)
print(x_k)
plt_approximate_processing2(approximate_values, is_show=False)
plt.show()
