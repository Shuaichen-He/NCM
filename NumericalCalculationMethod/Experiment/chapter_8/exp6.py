# -*- coding: UTF-8 -*-
"""
@file_name: exp6.py
@time: 2023-02-11
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from solving_equation_08.successive_compression_newton_method import SuccessiveCompressionNewton

p_coeff = np.array([1, 4, 2, -4, -3])
scn = SuccessiveCompressionNewton(p_coeff, 1e-16, 10000)
scn.fit_root()
scn.plt_polynomial_root()
print("多项式的全部零点和精度：", )
for i in range(len(scn.root)):
    print("%.15f, %.15e" % (scn.root[i], scn.precision[i]))