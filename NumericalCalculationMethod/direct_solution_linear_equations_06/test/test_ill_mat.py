# -*- coding: UTF-8 -*-
"""
@file_name: test_ill_mat.py
@time: 2021-11-26
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import sympy
from util_font import *

p = sympy.symbols("p", positive=True)  # 复合变量，假设为正数
A = sympy.Matrix([[1, sympy.sqrt(p)], [1, 1/sympy.sqrt(p)]]) # 构造符号矩阵
b = sympy.Matrix([1, 2])  # 符号右端向量
x_sym_sol = sympy.simplify(A.solve(b))  # 求解并简化，解析解
print(x_sym_sol)

# 求解方程组的数值解
Acond = A.condition_number().simplify()    # 条件数

# Function for solving numerically
AA = lambda p: np.array([[1, np.sqrt(p)], [1, 1/np.sqrt(p)]])
bb = np.array([1, 2])
x_num_sol = lambda p: np.linalg.solve(AA(p), bb)

# Graph the difference between the symbolic (exact) and numerical results.
p_vec = np.linspace(0.9, 1.1, 200)

fig, axes = plt.subplots(1, 2, figsize=(16, 4))

for n in range(2):
    x_sym = np.array([x_sym_sol[n].subs(p, pp).evalf() for pp in p_vec])
    x_num = np.array([x_num_sol(pp)[n] for pp in p_vec])
    axes[0].plot(p_vec, (x_num - x_sym)/x_sym, 'k')
axes[0].set_title(r"$Error_{sol} = \dfrac{numerical - symbolic}{symbolic} \qquad$", fontsize=18)
axes[0].set_xlabel(r'$p$', fontsize=18)
axes[0].set_ylabel(r'$Error$', fontsize=18)
axes[0].tick_params(labelsize=16)  # 刻度字体大小16
axes[1].plot(p_vec, [Acond.subs(p, pp).evalf() for pp in p_vec])
axes[1].set_title("$Condition \quad number$", fontsize=18)
axes[1].set_xlabel(r'$p$', fontsize=18)
axes[1].set_ylabel(r'$Value$', fontsize=18)
axes[1].tick_params(labelsize=16)  # 刻度字体大小16
plt.show()