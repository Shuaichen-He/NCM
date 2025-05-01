# -*- coding: UTF-8 -*-
"""
@file_name: exp4.py
@time: 2023-02-10
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import sympy
import matplotlib.pyplot as plt
from function_approximation_03.pade_rational_fraction_approximation import PadeRationalFractionApproximation

t = sympy.Symbol("t")
fun = sympy.sin(3 * t) * sympy.log(1 + 0.5 * t)

plt.figure(figsize=(18, 4))
orders = [6, 10, 15]
for i, order in enumerate(orders):
    plt.subplot(131 + i)
    prfa = PadeRationalFractionApproximation(fun, order=order)
    prfa.fit_rational_fraction()
    print(prfa.rational_fraction)
    if i == 0:
        prfa.plt_approximate(0, 2, is_show=False, is_fh_marker=False)
    else:
        prfa.plt_approximate(0, 2, is_show=False, is_fh_marker=True)
plt.show()