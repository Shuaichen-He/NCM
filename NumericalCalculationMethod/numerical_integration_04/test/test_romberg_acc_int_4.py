# -*- coding: UTF-8 -*-
"""
@file:test_romberg_acc_int_4.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import time
from numerical_integration_04.romberg_acceleration_quad import RombergAccelerationQuadrature
from util_font import *


int_fun = lambda x: x ** (3 / 2)  # 被积函数，例4
raq = RombergAccelerationQuadrature(int_fun, int_interval=[0, 1], accelerate_num=6)
raq.fit_int()  # 计算龙贝格积分
print(raq.Romberg_acc_table)  # 打印外推计算过程表
print(raq.int_value)  # 打印最终积分值

# 针对例3示例代码
int_fun2 = lambda x: np.exp(x ** 2) * ((0 <= x) & (x <= 2)) + \
                     80 / (4 - np.sin(16 * np.pi * x)) * ((2 < x) & (x <= 4))  # 例3

accelerate_num = np.arange(6, 26, 1)
int_error = np.zeros((len(accelerate_num), 2))
for i, an in enumerate(accelerate_num):
    raq = RombergAccelerationQuadrature(int_fun2, int_interval=[0, 4], accelerate_num=an)
    start = time.time()
    raq.fit_int()
    end = time.time()
    int_error[i, 0] = np.abs(raq.int_value - 57.764450125048512)
    int_error[i, 1] = end - start
    print("外推次数：%d，积分值：%.15f， 误差：%.10e，运行消耗时间：%.10fs" %
          (an, raq.int_value, int_error[i, 0], end - start))

fig = plt.figure(figsize=(14, 5))
plt.subplot(121)
plt.semilogy(accelerate_num, int_error[:, 0], "o-", lw=1.5, label="$\epsilon$")
idx = np.argmin(int_error[:, 0])
plt.semilogy(accelerate_num[idx], int_error[idx, 0], "D",
             label="$k=%d,\ \epsilon=%.5e$" % (accelerate_num[idx], int_error[idx, 0]))
plt.ylabel(r"$\epsilon=\vert I - I^* \vert$", fontdict={"fontsize": 18})
plt.xlabel("$k$", fontdict={"fontsize": 18})
plt.title("龙贝格求积随着外推次数$k$的增加绝对误差收敛曲线", fontdict={"fontsize": 18})
plt.tick_params(labelsize=16)  # 刻度字体大小16
plt.legend(frameon=False, fontsize=18)
plt.grid(ls=":")
plt.subplot(122)
plt.semilogy(accelerate_num, int_error[:, 1], "go-", lw=1.5, label="$Time$")
plt.ylabel("$Time(s)$", fontdict={"fontsize": 18})
plt.xlabel("$k$", fontdict={"fontsize": 18})
plt.title("龙贝格求积随着外推次数$k$的增加执行时间消耗曲线", fontdict={"fontsize": 18})
plt.tick_params(labelsize=16)  # 刻度字体大小16
plt.legend(frameon=False, fontsize=18)
plt.grid(ls=":")
plt.show()
