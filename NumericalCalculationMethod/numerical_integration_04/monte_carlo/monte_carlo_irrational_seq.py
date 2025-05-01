# -*- coding: UTF-8 -*-
"""
@file_name: monte_carlo_irrational_seq.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import math
import time
from util_font import *

# 构造线性独立的无理数，分别对应三重、四重和五重积分。
seq_3 = np.array([math.sqrt(2), math.sqrt(3), math.sqrt(6) / 3])
seq_4 = np.array([math.sqrt(2), math.sqrt(3), math.sqrt(6) / 3, math.sqrt(10)])
seq_5 = np.array([math.sqrt(2), math.sqrt(3), math.sqrt(6) / 3, math.sqrt(10),
                  math.sqrt(19)])


def generating_sequence(val, n):
    """
    生成无理数等分布序列函数，由于是指定无理数序列，故每次生成的序列一致
    :param val: 无理数序列，固定，无随机
    :param n: 尺度
    :return:
    """
    sequence = np.zeros((len(val), n))
    for i in range(n):
        sequence[:, i] = (i + 1) * val
    return sequence.T


def rnd_monte_carlo(n):
    """
    基本蒙特卡洛积分
    :param n: 随机数的数量
    :return:
    """
    X = np.zeros((n, 3))  # 存储均匀分布的随机数
    X[:, 0] = -1 + np.random.rand(n) * (1 - (-1))  # x1的上下限
    X[:, 1] = -1 + np.random.rand(n) * (1 - (-1))  # x2的上下限
    X[:, 2] = 0 + np.random.rand(n) * (1 - 0)  # x3的上下限
    # 查询各变量满足上下限的值，ind为布尔数组
    ind = (X[:, 1] >= - np.sqrt(1 - X[:, 0] ** 2)) & \
          (X[:, 1] <= np.sqrt(1 - X[:, 0] ** 2)) & \
          (X[:, 2] >= np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2)) & (X[:, 2] <= 1)
    return (1 - (-1)) * (1 - (-1)) * (1 - 0) * np.sum(int_fun(X[ind, :])) / n


def irrational_seq_monte_carlo(n):
    """
    等分布序列蒙特卡洛法
    :param n: 随机数的数量
    :return:
    """
    seq_data = generating_sequence(seq_3, n)  # 生成无理数
    X_01 = np.mod(seq_data, 1)  # [0, 1]区间
    X = np.zeros((n, 3))  # 存储满足上下限的数据
    X[:, 0] = -1 + X_01[:, 0] * (1 - (-1))  # x1的上下限
    X[:, 1] = -1 + X_01[:, 1] * (1 - (-1))  # x2的上下限
    X[:, 2] = 0 + X_01[:, 2] * (1 - 0)  # x3的上下限
    # 查询各变量满足上下限的值，ind为布尔数组
    ind = (X[:, 1] >= - np.sqrt(1 - X[:, 0] ** 2)) & \
          (X[:, 1] <= np.sqrt(1 - X[:, 0] ** 2)) & \
          (X[:, 2] >= np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2)) & (X[:, 2] <= 1)
    return (1 - (-1)) * (1 - (-1)) * (1 - 0) * np.sum(int_fun(X[ind, :])) / n


# 例15求解代码：
int_fun = lambda X: np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2)
num_x = np.arange(10000, 2000000, 10000)
int_res_1 = np.zeros(len(num_x))  # 存储每次模拟的近似积分值
int_res_2 = np.zeros(len(num_x))  # 存储每次模拟的近似积分值
start = time.time()
for i, n in enumerate(num_x):
    print(n)
    int_res_1[i] = np.pi / 6 - rnd_monte_carlo(n)  # 基本蒙特卡洛
    int_res_2[i] = np.pi / 6 - irrational_seq_monte_carlo(n)  # 等分布序列蒙特卡洛法

print(int_res_1[-1], int_res_2[-1])

plt.figure(figsize=(7, 5))
plt.plot(num_x, int_res_1, "k--", lw=1.5, label="随机序列")
plt.plot(num_x, int_res_2, "r-", lw=2, label="等分布序列")
plt.xlabel("$Random \quad numbers$", fontdict={"fontsize": 18})
plt.ylabel(r"$\epsilon= I - I^*$", fontdict={"fontsize": 18})
plt.tick_params(labelsize=16)  # 刻度字体大小16
plt.title(r"两种蒙特卡洛积分法的误差精度$\epsilon$曲线", fontdict={"fontsize": 18})
plt.grid(ls=":")
plt.legend(frameon=False, fontsize=18)
plt.show()


# 例16求解代码：
int_fun = lambda X: np.sqrt(X[:, 0] * X[:, 1]) * np.log(X[:, 2]) + \
                    np.sin(X[:, 3] / X[:, 1])
n = 10000000
seq_data = generating_sequence(seq_4, n)  # 生成无理数
X_01 = np.mod(seq_data, 1)  # [0, 1]区间
X = np.zeros((n, 4))  # 存储满足上下限的数据
X[:, 0] = 1 + X_01[:, 0] * (2 - 1)  # x1的上下限
X[:, 1] = 1 + X_01[:, 1] * (6 - 1)  # x2的上下限
X[:, 2] = 1 + X_01[:, 2] * (24 - 1)  # x3的上下限
X[:, 3] = 2 + X_01[:, 3] * (98 - 2)  # x4的上下限
# 查询各变量满足上下限的值，ind为布尔数组
ind = (X[:, 1] >= X[:, 0]) & (X[:, 1] <= 3 * X[:, 0]) & \
      (X[:, 2] >= X[:, 0] * X[:, 1]) & (X[:, 2] <= 2 * X[:, 0] * X[:, 1]) & \
      (X[:, 3] >= X[:, 0] + X[:, 0] * X[:, 2]) & \
      (X[:, 3] <= X[:, 0] + 2 * X[:, 0] * X[:, 2])
int_res = (2 - 1) * (6 - 1) * (24 - 1) * (98 - 2) * np.sum(int_fun(X[ind, :])) / n
print("积分近似值：", int_res)
