# -*- coding: UTF-8 -*-
"""
@author:Lenovo
@file:test_piecewise_cubic_hermite.py
@time:2021/08/30
"""
import numpy as np
from interpolation_02.piecewise_cubic_hermite_interpolation import PiecewiseCubicHermiteInterpolation


if __name__ == '__main__':
    x = np.linspace(0, 2 * np.pi, 10, endpoint=True)
    y = np.sin(x)  # 正弦函数模拟
    dy = np.cos(x)
    x0 = np.array([2.6, 4.0, 4.8])

    hi = PiecewiseCubicHermiteInterpolation(x, y, dy)
    hi.fit_interp()
    print(hi.poly_coefficient)
    y0 = hi.predict_x0(x0)
    print("插值点的值：", y0, "，精确值：", np.sin(x0))
    hi.plt_interpolation(x0, y0)
