# -*- coding: UTF-8 -*-
"""
@file_name: exp3.py
@time: 2023-02-11
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import sympy
from Experiment.util_font import *
from solving_equation_08.newton_root_method import NewtonRootMethod

t = sympy.Symbol("t")
equ = 2 * sympy.exp(-t) * sympy.sin(t)
x_init = [0.5, 3.0]  #  迭代初值

plt.figure(figsize=(14, 5))
line_style = ["*:", "o--", "+-.", "p-"]
method_vector = ["newton", "halley", "downhill", "multiroot"]
for i, x0 in enumerate(x_init):
    plt.subplot(121 + i)
    for style, method in zip(line_style, method_vector):
        newton = NewtonRootMethod(equ, x0=x0, eps=1e-16, method=method)
        newton.fit_root()
        iter_root_precision = np.asarray(newton.root_precision_info, np.float)
        plt.semilogy(iter_root_precision[:, 0], iter_root_precision[:, 2], style, lw=2,
                     label="$%s(k=%d, \ \epsilon=%.2e)$" %
                           (method, len(iter_root_precision), iter_root_precision[-1, 2]))
    plt.xlabel("$Iterations(k)$", fontdict={"fontsize": 18})
    plt.ylabel("$Precision(\epsilon)$", fontdict={"fontsize": 18})
    plt.title(r"各牛顿迭代法$(x_0=%.1f)$：$x_k$的$\epsilon=\vert f(x_k) \vert$收敛曲线" % x0,
              fontdict={"fontsize": 18})
    plt.legend(frameon=False, fontsize=16)
    plt.tick_params(labelsize=16)  # 刻度字体大小16
    plt.grid(ls=":")
plt.show()