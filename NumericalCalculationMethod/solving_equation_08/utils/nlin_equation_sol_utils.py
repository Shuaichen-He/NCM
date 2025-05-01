# -*- coding: UTF-8 -*-
"""
@file_name: nlin_equation_sol_entity_utils.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from util_font import *


class NLinearEquationSolUtils:
    """
    非线性方程求根，工具类定义
    """
    root = 0.0  # 方程的根

    # 参数的设置格式，字典形式
    #  options = {"eps": 1e-8, "display": "final", "pltFuns": False, "max_iter": 200}
    def __init__(self, fx, options=None):
        self.fx = fx  # 待求根方程
        self.options = options  # 参数信息，字典形式
        if self.options is None:
            self.eps = 1e-8  # 求解根的精度
            self.display = "display"  # 值有to_csv（存储外部文件），display（只显示最终结果）
            self.max_iter = 200  # 最大迭代次数
        else:
            self.eps = 1e-8 if self.options.get("eps") is None else self.options["eps"]
            self.display = "display" if self.options.get("display") is None else self.options["display"]
            self.plt_fun = False if self.options.get("pltFuns") is None else self.options["pltFuns"]
            self.max_iter = 200 if self.options.get("maxIter") is None else self.options["maxIter"]
        self.root_precision_info = None  # 近似根，精度，迭代次数等信息