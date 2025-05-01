# -*- coding: UTF-8 -*-
"""
@file_name: exp2.py
@time: 2023-02-10
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from numerical_differentiation_05.discrete_data_3_5_points_differentiation \
    import DiscreteData_3_5_PointsDifferentiation
from numerical_differentiation_05.discrete_data_cubic_spline_differetiation \
    import DiscreteDataCubicSplineDifferential
from numerical_differentiation_05.implicit_numerical_diff import ImplicitNumericalDifferentiation

# 离散数据形式的五点公式
x = np.linspace(0, 1, 11)
y = np.array([0.48, 0.38, 0.31, 0.33, 0.36, 0.41, 0.51, 0.43, 0.35, 0.29, 0.28])
ddpd5 = DiscreteData_3_5_PointsDifferentiation(x, y, points_type="five")
diff_val = ddpd5.cal_diff()
print("微分值：", diff_val)

# 三次样条插值, 修改边界条件，complete，natural
ddcsd = DiscreteDataCubicSplineDifferential(x, y, boundary_cond="complete")
diff_val = ddcsd.predict_diff_x0(x)
print("微分值：", diff_val)

# 隐式格式
imp_nd = ImplicitNumericalDifferentiation(x, y)
imp_nd.fit_diff()
diff_val = imp_nd.predict_x0(x)  # 近似值
print("微分值：", diff_val)
