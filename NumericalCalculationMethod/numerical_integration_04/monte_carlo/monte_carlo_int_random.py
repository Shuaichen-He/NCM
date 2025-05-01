# -*- coding: UTF-8 -*-
"""
@file_name: monte_carlo_int_random.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import time
import seaborn as sns
from util_font import *

# 例16
int_fun = lambda X: np.sqrt(X[:, 0] * X[:, 1]) * np.log(X[:, 2]) + np.sin(X[:, 3] / X[:, 1])
simulation_times, n = 1000, 1000000  # 模拟次数，以及每次模拟随机生成的随机数数量
int_res = np.zeros(simulation_times)  # 存储每次模拟的近似积分值
int_mean, int_std = np.zeros(simulation_times), np.zeros(simulation_times)
start = time.time()
for i in range(simulation_times):
    X = np.zeros((n, 4))  # 存储均匀分布的随机数
    X[:, 0] = 1 + np.random.rand(n) * (2 - 1)  # x1的上下限
    X[:, 1] = 1 + np.random.rand(n) * (6 - 1)  # x2的上下限
    X[:, 2] = 1 + np.random.rand(n) * (24 - 1)  # x3的上下限
    X[:, 3] = 2 + np.random.rand(n) * (98 - 2)  # x4的上下限
    # 查询各变量满足上下限的值，ind为布尔数组
    ind = (X[:, 1] >= X[:, 0]) & (X[:, 1] <= 3 * X[:, 0]) & \
          (X[:, 2] >= X[:, 0] * X[:, 1]) & (X[:, 2] <= 2 * X[:, 0] * X[:, 1]) & \
          (X[:, 3] >= X[:, 0] + X[:, 0] * X[:, 2]) & \
          (X[:, 3] <= X[:, 0] + 2 * X[:, 0] * X[:, 2])
    int_res[i] = (2 - 1) * (6 - 1) * (24 - 1) * (98 - 2) * int_fun(X[ind, :]).sum() / n
    int_mean[i], int_std[i] = np.mean(int_res[:i + 1]), np.std(int_res[:i + 1])

# 例15
# int_fun = lambda X: np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2)
#
# simulation_times, n = 2000, 100000  # 模拟次数，以及每次模拟随机生成的随机数数量
# int_res = np.zeros(simulation_times)  # 存储每次模拟的近似积分值
# int_mean, int_std = np.zeros(simulation_times), np.zeros(simulation_times)
# start = time.time()
# for i in range(simulation_times):
#     X = np.zeros((n, 3))  # 存储均匀分布的随机数
#     X[:, 0] = -1 + np.random.rand(n) * (1 - (-1))  # x1的上下限
#     X[:, 1] = -1 + np.random.rand(n) * (1 - (-1))  # x2的上下限
#     X[:, 2] = 0 + np.random.rand(n) * (1 - 0)  # x3的上下限
#     # 查询各变量满足上下限的值，ind为布尔数组
#     ind = (X[:, 1] >= - np.sqrt(1 - X[:, 0] ** 2)) & \
#           (X[:, 1] <= np.sqrt(1 - X[:, 0] ** 2)) & \
#           (X[:, 2] >= np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2)) & (X[:, 2] <= 1)
#     int_res[i] = (1 - (-1)) * (1 - (-1)) * (1 - 0) * np.sum(int_fun(X[ind, :])) / n
#     int_mean[i], int_std[i] = np.mean(int_res[:i + 1]), np.std(int_res[:i + 1])

# 如下为公共代码
end = time.time()
print("消耗时间：%.10f" % ((end - start) / simulation_times))
print("积分近似值：", int_mean[-1], "误差：", 1502.515542840579 - int_mean[-1])
# 可视化1000次积分的近似值
plt.figure(figsize=(14, 5))
plt.subplot(121)
plt.plot(range(simulation_times), int_mean, "r-", lw=1.5, label="$\mu$")
plt.plot(range(simulation_times), int_mean - int_std, "k--", lw=1, label="$\mu \pm \sigma$")
plt.plot(range(simulation_times), int_mean + int_std, "k--", lw=1)
plt.xlabel("$Random \ numbers$", fontdict={"fontsize": 18})
plt.ylabel("$Integral \ I^*$", fontdict={"fontsize": 18})
plt.tick_params(labelsize=16)  # 刻度字体大小16
plt.title(r"蒙特卡洛近似积分值：$%.10f(\pm%.5f)$" % (int_mean[-1], int_std[-1]), fontdict={"fontsize": 18})
plt.grid(ls=":")
plt.legend(frameon=False, fontsize=18, loc="upper right")
plt.tick_params(labelsize=16)  # 刻度字体大小16
plt.subplot(122)
plt.style.use("ggplot")
# 分别绘制直方图和核密度曲线
sns.distplot(int_res, bins=15, kde=False, hist_kws={'color': 'green'}, norm_hist=True)
sns.distplot(int_res, hist=False, kde_kws={"color": "red", 'linestyle': '-'},
             norm_hist=True)
plt.xlabel("$Bins$", fontdict={"fontsize": 18})
plt.ylabel("$Frequency$", fontdict={"fontsize": 18})
plt.title("近似积分值的直方图与核密度估计", fontdict={"fontsize": 18})
plt.tick_params(labelsize=16)  # 刻度字体大小16
plt.show()
