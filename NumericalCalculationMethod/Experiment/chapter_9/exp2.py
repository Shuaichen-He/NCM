# -*- coding: UTF-8 -*-
"""
@file_name: exp2.py
@time: 2023-02-11
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import sympy
from Experiment.util_font import *
import numpy as np
from nonlinear_equations_09.nlinequs_newton import NLinearFxNewton

x, y = sympy.symbols("x, y")
nlin_equs = [3 * x - sympy.cos(x) - sympy.sin(y), 4 * y - sympy.sin(x) - sympy.cos(y)]  # 定义的非线性方程组

# 可视化方法
plt.figure(figsize=(7, 5))
p0 = sympy.plot_implicit(nlin_equs[0], show=False, line_color="r", points=500)  # 绘制第一个方程
p1 = sympy.plot_implicit(nlin_equs[1], show=False, line_color="b", points=500)  # 绘制第二个方程
p0.extend(p1)  # 在方程1的基础上添加方程2的图像
p0.show()

# 牛顿法求解
x0 = np.array([2, 2])
mni = NLinearFxNewton(nlin_equs, [x, y], x0, max_iter=200, eps=1e-15, method="downhill", is_plt=True)
mni.fit_roots()
print(mni.iter_roots_precision[-1])
print(mni.fxs_precision)
print("下山因子：", mni.downhill_lambda)
print("=" * 80)