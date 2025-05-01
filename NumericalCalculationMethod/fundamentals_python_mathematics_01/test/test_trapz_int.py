# -*- coding: UTF-8 -*-
"""
@file_name: test_trapz_int.py
@time: 2022-10-31
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import time
from fundamentals_python_mathematics_01.trapezoidal_integral import TrapezoidalIntegral
from util_font import *

fh = lambda x: x ** 2 * np.sqrt(1 - x ** 2)
# fh = lambda x: np.exp(2 * x ** 2)  # 不可积问题
time1_list, time2_list = [], []
for i in range(100):
    start = time.time()
    t_int = TrapezoidalIntegral(fh, 0, 1, eps=1e-8, max_split_interval_num=200000)
    int_val, split_num = t_int.cal_trapezoid_int_vectorization()
    end = time.time()
    time1 = end - start
    time1_list.append(time1)
    if i == 99:
        print("划分区间数：", split_num)
        print("误差：", np.pi / 16 - int_val)
        t_int.plt_approximate_processing()

    # 非矢量化计算
    start = time.time()
    t_int = TrapezoidalIntegral(fh, 0, 1, eps=1e-8, max_split_interval_num=200000)
    int_val, split_num = t_int.cal_trapezoid_int_nvectorization()
    end = time.time()
    time2 = end - start
    time2_list.append(time2)
    if i == 99:
        print("划分区间数：", split_num)
        print("误差：", np.pi / 16 - int_val)
        t_int.plt_approximate_processing()


print("非矢量计算平均时间与矢量计算平均时间分别为：", np.mean(time2_list), np.mean(time1_list))
p = np.mean(time2_list) / np.mean(time1_list)
print("非矢量计算时间与矢量计算比值：", np.mean(time2_list) / np.mean(time1_list))
plt.figure(figsize=(7, 5))
plt.plot(time1_list, ls="-", label="矢量化, 平均: $%.5f$" % np.mean(time1_list))
plt.plot(time2_list, ls="--", label="非矢量化, 平均: $%.5f$" % np.mean(time2_list))
plt.xlabel("试验次数", fontsize=18)
plt.ylabel("消耗时间", fontsize=18)
plt.title("非矢量化与矢量计算消耗时间: 比值$%.2f$" % p, fontsize=20)
plt.tick_params(labelsize=18)
plt.legend(frameon=True, fontsize=18)
plt.show()