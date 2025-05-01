# -*- coding: UTF-8 -*-
"""
@file_name: test_exp3_2.py
@time: 2023-02-09
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import time
from Experiment.chapter_1.exp3_2 import *

fh = lambda x: np.exp(x)  # 被积函数
time1_list, time2_list = [], []  # 矢量化和非矢量化计算时间存储
for i in range(100):
    start = time.time()
    sol = cal_trapezoid_int_vectorization(fh, 0, 0.5, eps=1e-8, max_split_interval_num=200000)
    int_val, split_num, approximate_values = sol[0], sol[1], sol[2]
    end = time.time()
    time1 = end - start
    time1_list.append(time1)
    if i == 99:
        print("划分区间数：%d，积分近似值：%.15f" % (split_num, int_val))
        print("误差：%.10e" % abs(np.exp(0.5) - 1 - int_val))
        plt_approximate_processing(fh, approximate_values)  # 可视化

    # 非矢量化计算
    start = time.time()
    sol = cal_trapezoid_int_nvectorization(fh, 0, 0.5, eps=1e-8, max_split_interval_num=200000)
    int_val, split_num, approximate_values = sol[0], sol[1], sol[2]
    end = time.time()
    time2 = end - start
    time2_list.append(time2)
    if i == 99:
        print("划分区间数：%d，积分近似值：%.15f" % (split_num, int_val))
        print("误差：%.10e" % abs(np.exp(0.5) - 1 - int_val))
        plt_approximate_processing(fh, approximate_values)

print("非矢量计算平均时间与矢量计算平均时间分别为：", np.mean(time2_list), np.mean(time1_list))
print("非矢量计算时间与矢量计算比值：", np.mean(time2_list) / np.mean(time1_list))
