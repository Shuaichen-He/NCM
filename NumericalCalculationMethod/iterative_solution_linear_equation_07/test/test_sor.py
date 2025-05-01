# -*- coding: UTF-8 -*-
"""
@file_name: test_sor.py
@time: 2021-10-03
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import matplotlib.pyplot as plt
from iterative_solution_linear_equation_07.SOR_iterative import SORIteration


# A = np.array([[0.98, -0.05, -0.02], [-0.04, -0.9, 0.07], [-0.02, 0.09, 0.94]])
# b = np.array([1, 1, 1])
# A = np.array([[8, -3, 2], [4, 11, -1], [6, 3, 12]])
# b = np.array([20, 33, 36])
# A = np.array([[7, -1, 5], [3, -9, -2], [5, -4, 8]])  # 系数矩阵
# b = np.array([-20, -13, -22])  # 右端向量
# x0 = np.array([0, 0, 0])  # 初始解向量
# A = np.array([[28, -3, 0, 0, 0], [-3, 38, -10, 0, -5], [-10, 0, 25, -15, 0],
#               [0, 0, -15, 45, 0], [0, -5, 0, 0, 30]])  # 系数矩阵
# print(A)
# b = np.array([6, 5, 1, 4, 2])  # 右端向量

A = np.array([[-4, 1, 1, 1], [1, -4, 1, 1], [1, 1, -4, 1], [1, 1, 1, -4]])
b = np.array([1, 1, 1, 1])
x0 = np.array([0, 0, 0, 0])  # 初始解向量
omega_vector = np.linspace(0.1, 2, 19, endpoint=False)
object_, iter_num = [], []  # 存储对象，迭代次数
for omega in omega_vector:
    sor = SORIteration(A, b, x0, eps=1e-15, omega=omega, is_out_info=True)
    sor.fit_solve()  # SOR迭代求解
    object_.append(sor)
    iter_num.append(sor.iterative_info["Iteration_number"])
    print("=" * 60)

plt.figure(figsize=(14, 5))
plt.subplot(121)
idx = int(np.argmin(iter_num))
plt.plot(omega_vector, iter_num, "o--", lw=2)
plt.plot(omega_vector[idx], iter_num[idx], "D",
         label="$\omega=%.2f, \ k=%.d$" % (omega_vector[idx], iter_num[idx]))
plt.xlabel(r"$\omega \in [0.1, 1.9], \ step=0.1$", fontdict={"fontsize": 18})
plt.ylabel("$Iterations(k)$", fontdict={"fontsize": 18})
plt.title("不同松弛因子的$SOR$迭代法所需迭代次数", fontdict={"fontsize": 18})
plt.tick_params(labelsize=16)  # 刻度字体大小16
plt.legend(frameon=False, fontsize=18)
plt.grid(ls=":")
plt.subplot(122)
idx = int(np.argmin(iter_num))
object_[idx].plt_convergence(is_show=False)
plt.title("$SOR$的$\epsilon=\Vert b - Ax^* \Vert _2$收敛曲线：$\omega=%.2f$"
          % omega_vector[idx], fontdict={"fontsize": 18})
plt.show()
print(object_[idx].x)