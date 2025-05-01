# -*- coding: UTF-8 -*-
"""
@file_name: plt_Implicit_fun.py
@time: 2022-11-13
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import sympy
import matplotlib.pyplot as plt

plt.figure(figsize=(7, 5))
x, y = sympy.symbols("x, y")
# nlin_equs = [x ** 2 * sympy.exp(-x * y ** 2 / 2) + sympy.exp(-x / 2) * sympy.sin(x * y),
#              y ** 2 * sympy.cos(x + y ** 2) + x ** 2 * sympy.exp(x + y)]
nlin_equs = [x ** 3 + y ** 3 - 6 * x + 3, x ** 3 - y ** 3 - 6 * y + 2]
p0 = sympy.plot_implicit(nlin_equs[0], show=False, line_color="r", lw=2)
p1 = sympy.plot_implicit(nlin_equs[1], show=False, line_color="k", lw=2)
p0.extend(p1)
p0.show()
