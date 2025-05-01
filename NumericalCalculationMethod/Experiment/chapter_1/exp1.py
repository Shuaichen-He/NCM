# -*- coding: UTF-8 -*-
"""
@file_name: exp1.py
@time: 2023-02-09
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import math
import time
from decimal import Decimal, getcontext

getcontext().prec = 100  # 高精度计算，设置显示精度为100位


def calculate_pi(k):
    """
    拉马努金圆周率pi公式，计算给定项数的pi近似值
    :param k: 计算圆周率所需的累加次数k
    :return:
        approximate_pi，即满足项数k的pi近似值
    """
    pi = Decimal(0.0)  # 圆周率变量，初始化为0.0
    approximate_pi = Decimal(0.0)  # 初始化最终计算的近似圆周率值
    coef = Decimal(2 * math.sqrt(2) / 9801)  # 拉马努金公式系数
    # ===============================================================
    # 对于每一项，分别按照拉马努金公式循环计算k此，然后累加求解近似值
    # 其中math.factorial()为阶乘函数
    # Decimal()为高精度计算类
    # 打印输出每次计算后的近似值
    # ===============================================================
    for i in range(k):
        # 拉马努金计算公式
        pi += Decimal(math.factorial(4 * i) / math.factorial(i) ** 4) * \
              (1103 + 26390 * i) / Decimal(396 ** (4 * i))
        approximate_pi = Decimal(1 / pi / coef)  # 与系数乘积
    return approximate_pi


# 定义主函数，并调用函数，也可不使用主函数。

if __name__ == '__main__':
    k = int(input("请输入计算圆周率所需的累加次数 k = "))  # 接受用户输入且转换为整型
    start = time.time()  # 计算消耗时间，开始
    approximate_pi = calculate_pi(k)  # 调用函数，传递参数
    end = time.time()  # 计算消耗时间，结束
    print("消耗时间：%.15f" % (end - start))  # 消耗时间计算
    print(approximate_pi)  # 直接打印输出
    print("指定输出小数点后15位：%.15f" % approximate_pi)  # 占位符，格式化输出
