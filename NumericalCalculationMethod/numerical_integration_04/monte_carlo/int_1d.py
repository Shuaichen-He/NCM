# -*- coding: UTF-8 -*-
"""
@file_name: int_1d.py
@time: 2022-08-30
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from util_font import *

fh = lambda x: (np.cos(50 * x) + np.sin(20 * x)) ** 2  # 被积函数

a, b = 0, 1  # 积分上下限
num_x = np.arange(1000, 200000, 100)
int_res = np.zeros(len(num_x))  # 存储模拟积分近似值结果
int_mean = np.zeros(len(num_x))  # 存储迄今为止的积分均值
int_std = np.zeros(len(num_x))  # 存储迄今为止的积分标准方差
for i, n in enumerate(num_x):
    rnd_x = a + (b - a) * np.random.rand(n)  # 每次生成[a, b]之间的均匀随机数
    int_res[i] = (b - a) * np.sum(fh(rnd_x)) / n
    int_mean[i] = np.mean(int_res[:i + 1])
    int_std[i] = np.std(int_res[:i + 1])

true_int = np.cos(30)/30 - np.cos(70)/70 - np.sin(40)/80 + np.sin(100)/200 + 103/105  # 精确值
print("积分近似值为%.15f， 误差为%.15e" % (int_mean[-1], true_int - int_mean[-1]))

plt.figure(figsize=(14, 5))
plt.subplot(121)
xi = np.linspace(0, 1, 500)
plt.plot(xi, fh(xi), "k-", lw=1.5)
plt.xlabel("$x$", fontdict={"fontsize": 18})
plt.ylabel("$f(x)$", fontdict={"fontsize": 18})
plt.tick_params(labelsize=16)  # 刻度字体大小16
plt.title("$f(x)=(cos50x + sin20x)^2$", fontdict={"fontsize": 18})
plt.grid(ls=":")
plt.subplot(122)
plt.plot(num_x, int_mean, "r-", lw=1.5, label="$\mu$")
plt.plot(num_x, int_mean - int_std, "k--", lw=1, label="$\mu \pm \sigma$")
plt.plot(num_x, int_mean + int_std, "k--", lw=1)
plt.xlabel("$Random \ numbers$", fontdict={"fontsize": 18})
plt.ylabel("$\int_{0}^{1}f(x)dx$", fontdict={"fontsize": 18})
plt.tick_params(labelsize=16)  # 刻度字体大小16
plt.title(r"蒙特卡洛近似计算$\int_{0}^{1}f(x)dx \approx %.5f(\pm%.5f)$" %
          (int_mean[-1], int_std[-1]), fontdict={"fontsize": 18})
plt.grid(ls=":")
plt.legend(frameon=False, fontsize=18)
plt.show()
