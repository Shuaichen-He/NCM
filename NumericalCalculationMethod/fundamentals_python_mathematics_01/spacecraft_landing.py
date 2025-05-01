# -*- coding: UTF-8 -*-
"""
@file_name: spacecraft_landing.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from util_font import *

G = 9.80665  # 重力加速度
# 控制反向喷射的函数：忽略飞船质量的变化，理想情况下模拟，假设反向喷射加速度a = G
retrofire = lambda t, rft, C: -C * G if t >= rft else 0.0


def spacecraft_simulation(init_v, init_h, rft, th=0.01, C=1.0):
    """
    飞船软着陆模拟计算，即采用显示欧拉法求解一阶微分方程组
    :param init_v: 初始的速度
    :param init_h: 初始的高度
    :param rft: 反向喷射开始的时刻
    :param th: 时间跨度，时间步长
    :param C: 反向喷射加速度的系数
    :return:
    """
    height, v, t = init_h, init_v, 0  # 初始化飞船的高度、速度和时刻
    info_dict = dict()  # 计算过程信息存储，采用字典，包括时刻、高度和速度，每个均为列表
    info_dict["time"], info_dict["height"], info_dict["speed"] = [t], [height], [v]
    # 1. 自由落体运动，2. 反向喷射， 通过函数retrofire(t, rft, C)控制计算
    while 0 <= height <= init_h:  # 循环计算到软着陆或软着陆失败后道初始的高度为止
        t += th  # 更新时刻表
        v += (G + retrofire(t, rft, C)) * th  # 更新飞船的速度
        height -= v * th  # 更新飞船的位置（高度），注意为下降
        info_dict["time"].append(t)  # 对应字典的键存储值，每个键所对应的值为列表结构
        info_dict["speed"].append(v)  # 速度
        info_dict["height"].append(height)  # 高度

    idx = int(rft / th)  # 计算反向喷射时刻的索引值，用于后续可视化
    ths_retrofire = [info_dict["time"][idx], info_dict["height"][idx], info_dict["speed"][idx]]
    return info_dict, ths_retrofire


def plt_spacecraft_simulation(info_dict, rf_ths, i=0):
    """
    绘制飞船软着陆下降轨迹和速度变化曲线
    :param info_dict: 计算过程信息存储的字典
    :param rf_ths: 反向喷射的时刻、高度和速度
    """
    time, height, speed = info_dict["time"], info_dict["height"], info_dict["speed"]
    idx = np.argmin(height)  # 飞船软着陆过程中最小的高度值索引
    time_min, height_min, speed_min = time[idx], height[idx], speed[idx]
    # plt.figure(figsize=(14, 5))  # 若独立绘图，可取消注释；绘制子图，则注释
    plt.subplot(221 + 2 * i)
    plt.plot(time, height, "k-", lw=1.5, label="轨迹曲线")
    plt.plot(rf_ths[0], rf_ths[1], "ro", label="反向喷射点")
    plt.plot(time_min, height_min, "bs", label="最小高度：$%.2f$" % abs(height_min))
    plt.xlabel("时间", fontdict={"fontsize": 18})
    plt.ylabel("高度", fontdict={"fontsize": 18})
    if i == 0:
        plt.title("模拟飞船软着陆问题轨迹曲线（成功）", fontdict={"fontsize": 18})
    else:
        plt.title("模拟飞船软着陆问题轨迹曲线（失败）", fontdict={"fontsize": 18})
    plt.legend(frameon=False, fontsize=16, loc="best")  # 添加图例
    plt.tick_params(labelsize=16)  # 刻度字体大小16
    plt.grid(ls=":")
    plt.subplot(221 + 2 * i + 1)
    plt.plot(time, np.abs(speed), "k-", lw=1.5, label="速度曲线")
    plt.plot(rf_ths[0], rf_ths[2], "ro", label="反向喷射点")
    plt.plot(time_min, speed_min, "bs", label="最小速度：$%.2f$" % speed_min)
    plt.xlabel("时间", fontdict={"fontsize": 18})
    plt.ylabel("速度", fontdict={"fontsize": 18})
    if i == 0:
        plt.title("模拟飞船软着陆问题速度曲线（成功）", fontdict={"fontsize": 18})
    else:
        plt.title("模拟飞船软着陆问题速度曲线（失败）", fontdict={"fontsize": 18})
    plt.legend(frameon=False, fontsize=16, loc="best")  # 添加图例
    plt.tick_params(labelsize=16)  # 刻度字体大小16
    plt.grid(ls=":")
    # plt.show() # 若独立绘图，可取消注释；


if __name__ == '__main__':
    # 既有位置参数又有关键字参数，则位置参数必须在关键字参数之前传参
    ths_dict, ths = spacecraft_simulation(0, 1000, rft=7.987, th=0.001, C=1.455)  # 位置参数 + 关键字参数
    plt.figure(figsize=(12, 8))
    plt_spacecraft_simulation(ths_dict, ths)
    ths_dict, ths = spacecraft_simulation(0, 1000, rft=7.5, th=0.001, C=1.455)  # 位置参数 + 关键字参数
    plt_spacecraft_simulation(ths_dict, ths, i=1)
    plt.show()
