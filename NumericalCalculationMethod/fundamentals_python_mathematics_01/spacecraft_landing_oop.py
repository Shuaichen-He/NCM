# -*- coding: UTF-8 -*-
"""
@file_name: spacecraft_landing_oop.py
@time: 2022-10-30
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from util_font import *


class SpaceCraftLandingSimulation:
    """
    面向对象设计：飞船软着陆模拟计算，即采用显示欧拉法求解一阶微分方程组
    """
    G = 9.80665  # 类实例，重力加速度，不同的对象间可共享值

    def __init__(self, init_v, init_h, rf_time, th=0.01, C=1.5):
        self.init_v = init_v  # 初始的速度
        self.init_h = init_h  # 初始的高度
        self.rf_time = rf_time  # 反向喷射开始的时刻
        self.th = th  # 时间跨度，时间步长
        self.C = C  # 反向喷射加速度的系数
        self.info_dict = dict()  # 计算过程信息存储，采用字典，包括时刻、高度和速度，每个均为列表
        self.rf_ths = None  # 为飞船提供反向喷射加速度时的时刻、高度和速度，仅为可视化

    # 控制反向喷射的函数：忽略飞船质量的变化，理想情况下模拟，假设反向喷射加速度a = G
    retrofire = lambda self, t: -self.C * self.G if t >= self.rf_time else 0.0

    def simulate_cal(self):
        """
        核心算法：飞船软着陆模拟计算
        :return:
        """
        height, v, t = self.init_h, self.init_v, 0  # 初始化飞船的高度、速度和时刻
        self.info_dict["time"] = [t]  # 时刻
        self.info_dict["height"] = [height]  # 高度
        self.info_dict["speed"] = [v]  # 速度
        # 1. 自由落体运动，2. 反向喷射， 通过实例函数retrofire(t)控制计算
        while 0 <= height <= self.init_h:  # 循环计算到软着陆或初始的高度为止
            t += self.th  # 更新时刻表
            v += (self.G + self.retrofire(t)) * self.th  # 更新飞船的速度
            height -= v * self.th  # 更新飞船的位置（高度），注意为下降
            self.info_dict["time"].append(t)  # 对应字典的键存储值，每个键所对应的值为列表结构
            self.info_dict["speed"].append(v)  # 速度
            self.info_dict["height"].append(height)  # 高度

        idx = int(self.rf_time / self.th)  # 计算反向喷射时刻的索引值，用于后续可视化
        self.rf_ths = [self.info_dict["time"][idx], self.info_dict["height"][idx], self.info_dict["speed"][idx]]

    def plt_simulation_processing(self):
        """
        绘制飞船软着陆下降轨迹和速度变化曲线
        :return:
        """
        idx = np.argmin(self.info_dict["height"])  # 飞船软着陆过程中最小的高度值索引
        time_min = self.info_dict["time"][idx]
        height_min, speed_min = self.info_dict["height"][idx], self.info_dict["speed"][idx]
        plt.figure(figsize=(12, 4.5))
        plt.subplot(121)
        plt.plot(self.info_dict["time"], self.info_dict["height"], "k-", lw=1.5, label="轨迹曲线")
        plt.plot(self.rf_ths[0], self.rf_ths[1], "ro", label="反向喷射点")
        plt.plot(time_min, height_min, "bs", label="最小高度：$%.2f$" % height_min)
        plt.xlabel("时间", fontdict={"fontsize": 18})
        plt.ylabel("高度", fontdict={"fontsize": 18})
        plt.title("模拟飞船软着陆问题轨迹曲线", fontdict={"fontsize": 18})
        plt.legend(frameon=False, fontsize=16, loc="best")  # 添加图例
        plt.tick_params(labelsize=16)  # 刻度字体大小16
        plt.grid(ls=":")
        plt.subplot(122)
        plt.plot(self.info_dict["time"], np.abs(self.info_dict["speed"]), "k-", lw=1.5, label="速度曲线")
        plt.plot(self.rf_ths[0], self.rf_ths[2], "ro", label="反向喷射点")
        plt.plot(time_min, speed_min, "bs", label="最小速度：$%.2f$" % speed_min)
        plt.xlabel("时间", fontdict={"fontsize": 18})
        plt.ylabel("速度", fontdict={"fontsize": 18})
        plt.title("模拟飞船软着陆问题速度曲线", fontdict={"fontsize": 18})
        plt.legend(frameon=False, fontsize=16, loc="best")  # 添加图例
        plt.tick_params(labelsize=16)  # 刻度字体大小16
        plt.grid(ls=":")
        plt.show()


if __name__ == '__main__':
    scls = SpaceCraftLandingSimulation(0, 1000, rf_time=7.99, th=0.001, C=1.455)  # 初始化对象，提供参数
    scls.simulate_cal()  # 模拟计算
    scls.plt_simulation_processing()  # 可视化
