# -*- coding: UTF-8 -*-
"""
@file_name: exp1.py
@time: 2023-02-11
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from iterative_solution_linear_equation_07.jacobi_gauss_seidel_iterative import JacobiGSlIterativeMethod
from iterative_solution_linear_equation_07.SOR_iterative import SORIteration
from Experiment.util_font import *  # 导入字体文件

A = np.array([[28, -3, 0, 0, 0], [-3, 38, -10, 0, -5], [-10, 0, 25, -15, 0],
              [0, 0, -15, 45, 0], [0, -5, 0, 0, 30]])  # 系数矩阵
b = np.array([6, 5, 1, 4, 2])  # 右端向量
x0 = np.array([0, 0, 0, 0, 0])  # 初始解向量

jacobi = JacobiGSlIterativeMethod(A, b, x0, eps=1e-15, method="Jacobi", is_out_info=True)
jacobi.fit_solve()  # 雅可比迭代求解
print("=" * 80)
guass_seidel = JacobiGSlIterativeMethod(A, b, x0, eps=1e-15, method="G-S", is_out_info=True)
guass_seidel.fit_solve()  # G-S迭代求解
plt.figure(figsize=(14, 5))
plt.subplot(121)
jacobi.plt_convergence(is_show=False)  # 可视化雅可比迭代收敛性
plt.subplot(122)
guass_seidel.plt_convergence(is_show=False)  # 可视化G-S迭代收敛性
plt.show()

plt.figure(figsize=(7, 5))
iter_num_jacobi = jacobi.iterative_info["Iteration_number"]  # 获取雅可比迭代次数
iter_num_jacobi = np.linspace(1, iter_num_jacobi, iter_num_jacobi)
iter_num_gs = guass_seidel.iterative_info["Iteration_number"]  # 获取G-S迭代次数
iter_num_gs = np.linspace(1, iter_num_gs, iter_num_gs)
plt.semilogy(iter_num_jacobi, jacobi.precision, "o--", lw=1.5,
             label=r"$Jacobi: \ \epsilon=%.3e,\  n_{iter}=%d$" % (jacobi.precision[-1], iter_num_jacobi[-1]))
plt.semilogy(iter_num_gs, guass_seidel.precision, "*-", lw=1.5,
             label="$GS: \ \epsilon=%.3e, \ n_{iter}=%d$" % (guass_seidel.precision[-1], iter_num_gs[-1]))  # 绘制G-S
plt.xlabel("$Iterations$", fontdict={"fontsize": 18})
plt.ylabel("$Precision(\epsilon)$", fontdict={"fontsize": 18})
plt.title("$Jacobi$和$Gauss-Seidel$的$\epsilon=\Vert b - Ax^* \Vert _2$收敛曲线", fontdict={"fontsize": 18})
plt.legend(frameon=False, fontsize=16)
plt.tick_params(labelsize=16)  # 刻度字体大小16
plt.grid(ls=":")
plt.show()

# 超松弛迭代法
omega_vector = np.linspace(0.1, 2, 19, endpoint=False)
object_, iter_num = [], []  # 存储对象，迭代次数
for omega in omega_vector:
    sor = SORIteration(A, b, x0, eps=1e-15, omega=omega, is_out_info=False)
    sor.fit_solve()  # SOR迭代求解
    object_.append(sor)
    iter_num.append(sor.iterative_info["Iteration_number"])

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
print("=" * 80)
# 最佳超松弛因子下的迭代信息
for key in object_[idx].iterative_info.keys():
    print(key, ": ", object_[idx].iterative_info[key])
