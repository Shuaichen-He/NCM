# -*- coding: UTF-8 -*-
"""
@file_name: test_cubic_spline_simple.py
@time: 2023-11-15
@IDE: PyCharm  Python: 3.9.7
@copyright: 信阳师范大学, http://maths.xynu.edu.cn
"""
import numpy as np
from interpolation_02.cubic_spline_interpolation import CubicSplineInterpolation

# x = np.array([1, 2, 4, 5])
# y = np.array([1, 3, 4, 2])

# x = np.array([1.94, 1.95, 1.96, 1.97, 1.98, 1.99])
# y = np.array([132.165, 151.326, 179.323, 203.302, 226.542, 249.633])

# x = np.array([27.7, 28, 29, 30])
# y = np.array([4.1, 4.3, 4.1, 3.0])
# dy = np.array([3.0, -4.0])

x = np.array([0, 1, 2, 3, 4, 5, 6])
f = lambda x: np.sin(x)
df = lambda x: np.cos(x)
y = f(x)
dy = df(x)
# dy = np.array([3.0, -4.0])

# csi = CubicSplineInterpolation(x, y, dy, boundary_cond="complete")
csi = CubicSplineInterpolation(x, y,dy, boundary_cond="natural")
csi.fit_interp()
print(csi.poly_coefficient)
csi.plt_interpolation()
