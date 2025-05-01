# -*- coding: UTF-8 -*-
"""
@file_name: newton_diff_speed_distance.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""

import numpy as np
from interpolation_02.newton_difference_interpolation import NewtonDifferenceInterpolation

speed = np.linspace(20, 150, 14, endpoint=True)  # 时速(km/s)
speed_ = speed * 1000 / 3600  # 转化为m/s
reaction_dis = speed_ * 10  # 10s内的反应距离数据
distance = np.array([3.15, 7.08, 12.59, 19.68, 28.34, 38.57, 50.4, 63.75, 78.71,
                     95.22, 113.29, 132.93, 154.12, 176.87])  # 刹车距离数据
distance = reaction_dis + distance + 10  # 停车视距 = 反应 + 制动 + 安全（距离）

diff_method = ["forward", "backward"]  # 分别以牛顿前向和后向差分插值求解
vi = np.arange(20, 151, 1)  # 以1为步长，离散化速度区间数据，然后进行插值计算
for method in diff_method:
    # 第一问求解
    ndi_obj = NewtonDifferenceInterpolation(speed, distance, diff_method=method)  # 类实例化对象
    ndi_obj.fit_interp()  # 插值计算，生成多项式
    ndi_obj.plt_interpolation()  # 可视化速度与停车视距的关系多项式
    dist_i = ndi_obj.predict_x0(vi)  # 模拟计算各时速下的停车视距
    # 假设忽略安全距离（以不撞上障碍物为标准），以求得最高时速，
    dist_i0 = np.abs(dist_i - 120)  # 停车视距 - 120（制动距离）
    idx = np.argsort(dist_i0)  # 升序排列，获得第一个序号
    print(method + "时速最高不能超过：", vi[idx[0]], "m/s")

    # 第二问求解
    visual_distance = dist_i[vi == 125]  # 时速为自变量，近似查询对应有效视距即可
    print(method + "可视距离为：", visual_distance[0], "m")
