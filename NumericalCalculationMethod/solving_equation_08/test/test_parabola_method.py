# -*- coding: UTF-8 -*-
"""
@file:test_parabola_method.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from solving_equation_08.parabola_method import ThreePointParabolaMethod
from solving_equation_08.double_points_secant_method import DoublePointsSecantMethod

equation = lambda x: x * np.exp(x) - 1

pm = ThreePointParabolaMethod(equation, [0, 1], eps=1e-16)
pm.fit_root()
for i in range(len(pm.root_precision_info)):
    print("%.15f, %.15e" % (pm.root_precision_info[i, 1], pm.root_precision_info[i, 2]))
print("=" * 80)
sm = DoublePointsSecantMethod(equation, [0, 1], eps=1e-16)
sm.fit_root()
for i in range(len(sm.root_precision_info)):
    print("%.15f, %.15e" % (sm.root_precision_info[i, 1], sm.root_precision_info[i, 2]))