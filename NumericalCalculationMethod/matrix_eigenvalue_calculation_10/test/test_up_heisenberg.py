# -*- coding: UTF-8 -*-
"""
@file_name: test_up_heisenberg.py
@time: 2021-11-08
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from matrix_eigenvalue_calculation_10.up_heisenberg_matrix import UPHeisenbergMatrix

if __name__ == '__main__':
    A = np.array([[-4, -3, -7], [2, 3, 2], [4, 2, 7]])
    # A = np.array([[2, 3, 4, 5, 6], [4, 4, 5, 6, 7], [0, 3, 6, 7, 8], [0, 0, 2, 8, 9], [0, 0, 0, 1, 0]])
    heisenberg = UPHeisenbergMatrix(A)
    h_m = heisenberg.cal_heisenberg_mat()
    print(h_m)