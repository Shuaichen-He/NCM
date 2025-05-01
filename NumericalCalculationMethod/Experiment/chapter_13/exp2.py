# -*- coding: UTF-8 -*-
"""
@file_name: exp2.py
@time: 2023-02-12
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import sympy
from numerical_optimization_13.powell_method import PowellOptimization
from numerical_optimization_13.nelder_mead_2d import NelderMeadOptimization
from Experiment.util_font import *

# 等值线可视化
fh = lambda x, y: x * np.exp(-x ** 2 - y ** 2)
plt.figure(figsize=(7, 5))
x, y = np.linspace(-2, 2, 100), np.linspace(-2, 2, 100)
xi, yi = np.meshgrid(x, y)
zi = fh(xi, yi)
c = plt.contour(xi, yi, zi, levels=15, cmap=plt.get_cmap("jet"))
plt.clabel(c, inline=True, fontsize=10)  # 添加等高数值标记
plt.xlabel("$x$", fontdict={"fontsize": 18})
plt.ylabel("$y$", fontdict={"fontsize": 18})
plt.title("二元函数等值线图", fontdict={"fontsize": 18})
plt.grid(ls=":")
plt.tick_params(labelsize=18)
plt.show()

f_min = lambda x: x[0] * np.exp(-x[0] ** 2 - x[1] ** 2)  # 极小值
V_k = np.array([[-1.5, 0], [-0.5, 1], [-0.1, -1]])
eps = 1e-15
nmo = NelderMeadOptimization(f_min, V_k, eps, is_minimum=True)  # 此处修改is_minimum为False求极大值
m_x = nmo.fit_optimize()
print("[(%.15f, %.15f), %.15f]" % (m_x[0], m_x[1], m_x[2]))
nmo.plt_optimization([-2, 2], [-2, 2])

f_max = lambda x: -1 * (x[0] * np.exp(-x[0] ** 2 - x[1] ** 2))  # 极大值
V_k = np.array([[0.2, -1], [1, 1], [1.5, -0.5]])
nmo = NelderMeadOptimization(f_max, V_k, eps, is_minimum=False)  # 此处修改is_minimum为False求极大值
m_x = nmo.fit_optimize()
print("[(%.15f, %.15f), %.15f]" % (m_x[0], m_x[1], m_x[2]))
nmo.plt_optimization([-2, 2], [-2, 2])
print("=" * 80)

# 鲍威尔方法
x = sympy.symbols("x_1:3")
f_min = x[0] * sympy.exp(-x[0] ** 2 - x[1] ** 2)
x0, eps = [-1.5, 1], 1e-15
po = PowellOptimization(f_min, x0, eps, is_minimum=True)  # 极小值
e_x = po.fit_optimize()
po.plt_optimization([-2, 2], [-2, 2])
print("%.15f, %.15f" % (e_x[0], e_x[1]))
print(f_min.subs({x[0]: e_x[0], x[1]: e_x[1]}))

x = sympy.symbols("x_1:3")
f_max = -1 * x[0] * sympy.exp(-x[0] ** 2 - x[1] ** 2)
x0, eps = [1.5, 1], 1e-15
po = PowellOptimization(f_max, x0, eps, is_minimum=False)  # 极大值
e_x = po.fit_optimize()
po.plt_optimization([-2, 2], [-2, 2])
print("%.15f, %.15f" % (e_x[0], e_x[1]))
print(-1 * f_max.subs({x[0]: e_x[0], x[1]: e_x[1]}))