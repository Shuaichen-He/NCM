# -*- coding: UTF-8 -*-
"""
@file_name: exp3_1.py
@time: 2023-02-09
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import time
from fundamentals_python_mathematics_01.trapezoidal_integral import TrapezoidalIntegral

fh = lambda x: np.exp(x)  # 被积函数
time1_list, time2_list = [], []  # 矢量化和非矢量化计算时间存储
for i in range(100):
    start = time.time()
    t_int = TrapezoidalIntegral(fh, 0, 0.5, eps=1e-8, max_split_interval_num=200000)
    int_val, split_num = t_int.cal_trapezoid_int_vectorization()
    end = time.time()
    time1 = end - start
    time1_list.append(time1)
    if i == 99:
        print("划分区间数：%d，积分近似值：%.15f" % (split_num, int_val))
        print("误差：%.10e" % abs(np.exp(0.5) - 1 - int_val))
        t_int.plt_approximate_processing()

    # 非矢量化计算
    start = time.time()
    t_int = TrapezoidalIntegral(fh, 0, 0.5, eps=1e-8, max_split_interval_num=200000)
    int_val, split_num = t_int.cal_trapezoid_int_nvectorization()
    end = time.time()
    time2 = end - start
    time2_list.append(time2)
    if i == 99:
        print("划分区间数：%d，积分近似值：%.15f" % (split_num, int_val))
        print("误差：%.10e" % abs(np.exp(0.5) - 1 - int_val))
        t_int.plt_approximate_processing()
print("非矢量计算平均时间与矢量计算平均时间分别为：", np.mean(time2_list), np.mean(time1_list))
print("非矢量计算时间与矢量计算比值：", np.mean(time2_list) / np.mean(time1_list))