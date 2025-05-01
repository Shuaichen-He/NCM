# -*- coding: UTF-8 -*-
"""
@file_name: util_font.py
@time: 2023-02-11
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import matplotlib.pyplot as plt
import matplotlib as mpl

# 设置数学模式下的字体格式和中文显示
rc = {"font.family": "serif", "mathtext.fontset": "cm"}
plt.rcParams.update(rc)
mpl.rcParams["font.family"] = "FangSong"  # 中文显示
plt.rcParams["axes.unicode_minus"] = False  # 解决坐标轴负数的负号显示问题