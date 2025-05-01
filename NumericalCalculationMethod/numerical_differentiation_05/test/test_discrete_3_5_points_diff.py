# -*- coding: UTF-8 -*-
"""
@file_name: test_discrete_3_5_points_diff.py
@time: 2021-11-24
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from numerical_differentiation_05.discrete_data_3_5_points_differentiation \
    import DiscreteData_3_5_PointsDifferentiation

fun = lambda x: np.sin(x) * np.exp(-x)

x = np.linspace(1, 5, 10)
ddpd3 = DiscreteData_3_5_PointsDifferentiation(x, fun(x), points_type="three")
diff_val = ddpd3.cal_diff()
y_true = np.exp(-x) * (np.cos(x) - np.sin(x))
print("微分值：", diff_val, "\n精确值：", y_true, "\n误差：", y_true - diff_val)
# err = y_true - diff_val
# for i in range(10):
#     print("%.8f" % err[i])
ddpd5 = DiscreteData_3_5_PointsDifferentiation(x, fun(x), points_type="five")
diff_val = ddpd5.cal_diff()
y_true = np.exp(-x) * (np.cos(x) - np.sin(x))
print("微分值：", diff_val, "\n精确值：", y_true, "\n误差：", y_true - diff_val)
