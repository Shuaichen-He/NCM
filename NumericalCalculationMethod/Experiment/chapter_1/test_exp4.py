# -*- coding: UTF-8 -*-
"""
@file_name: test_exp4.py
@time: 2023-02-09
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import math
from Experiment.chapter_1.exp4 import IterationExp

IExp = IterationExp(x0=-100, eps=1e-16, max_iter=100)  # 初始化对象
x = IExp.cal_iter()  # 调用迭代方法，执行迭代
IExp.plt_approximate_processing()  # 可视化
print("近似值：%.15f， 绝对值误差：%.15e" % (x, abs((-1 + math.sqrt(5)) / 2 - x)))
