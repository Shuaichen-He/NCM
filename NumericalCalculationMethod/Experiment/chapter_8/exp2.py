# -*- coding: UTF-8 -*-
"""
@file_name: exp2.py
@time: 2023-02-11
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from solving_equation_08.iterative_solution_method import IterativeSolutionMethod_Root
from Experiment.util_font import *

fai_x = lambda x: 2 * np.log(x) + np.log(3)  # 迭代公式
x0 = 3.5  # 迭代初值

method_eval = ["stable", "aitken", "steffensen"]
plt.figure(figsize=(7, 5))
line_stype = ["*-.", "o--", "s-"]
for method, style in zip(method_eval, line_stype):
    ism = IterativeSolutionMethod_Root(fai_x, x0=x0, eps=1e-16, max_iter=200, method=method)
    ism.fit_root()
    info = ism.root_precision_info
    plt.semilogy(info[:, 0], info[:, 2], style,
                 label="$%s: \ k=%d, \ \epsilon=%.2e$" % (method, len(info), info[-1, 2]))
plt.xlabel("$Iterations(k)$", fontdict={"fontsize": 18})
plt.ylabel("$Precision(\epsilon)$", fontdict={"fontsize": 18})
plt.title(r"各迭代法近似根$x^*$的$\epsilon=\vert x_{k+1} - x_k \vert$收敛曲线",
          fontdict={"fontsize": 18})
plt.legend(frameon=False, fontsize=16)
plt.tick_params(labelsize=16)  # 刻度字体大小16
plt.grid(ls=":")
plt.show()