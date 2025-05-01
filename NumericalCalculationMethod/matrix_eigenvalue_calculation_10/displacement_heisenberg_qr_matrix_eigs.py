# -*- coding: UTF-8 -*-
"""
@file_name: displacement_qr_orthogonal_matrix_eigs.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import sympy
import math
from matrix_eigenvalue_calculation_10.heisenberg_qr_matrix_eigs import HeisenbergQRMatrixEig
from decimal import *

getcontext().prec = 128


class DisplacementHeisenbergQRMatrixEig(HeisenbergQRMatrixEig):
    """
    位移QR正交变换法，上海森伯格矩阵，默认Givens 正交化方法求解矩阵全部特征值
    """

    def __init__(self, A, eps=1e-8, max_iter=1000, displacement="Rayleigh",
                 transform="Givens"):
        HeisenbergQRMatrixEig.__init__(self, A, eps, max_iter, transform)
        self.displacement = displacement  # 位移方法，包括瑞利商rayleigh和威尔金斯Wilkins

    def fit_eig(self):
        """
        上海森伯格矩阵，QR方法求解矩阵全部特征值，重写父类方法
        :return:
        """
        orthogonal_fun = None  # 用于选择正交化的方法
        if self.transform.lower() == "givens":
            orthogonal_fun = eval("self._givens_rotation_")
        elif self.transform.lower() == "schmidt":
            orthogonal_fun = eval("self._schmidt_orthogonal_")
        elif self.transform.lower() == "householder":
            orthogonal_fun = eval("self._householder_transformation_")
        else:
            print("QR正交分解有误，支持Givens、Schmidt或Householder。")
            exit(0)
        orthogonal_mat, miu = np.copy(self.A), np.infty
        for i in range(self.max_iter):
            if self.displacement.lower() == "rayleigh":
                miu = orthogonal_mat[-1, -1]  # 位移
            elif self.displacement.lower() == "wilkins":
                miu = self._miu_displacement_wilkins_(orthogonal_mat)
            else:
                print("位移方法仅支持Rayleigh或Wilkins.")
                exit(0)
            Q, R = orthogonal_fun(orthogonal_mat - miu * np.eye(self.n))
            orthogonal_mat = np.dot(R, Q) + miu * np.eye(self.n)
            self.iter_eigenvalues.append(np.diag(orthogonal_mat))
            if len(self.iter_eigenvalues) > 1:
                prec = np.linalg.norm(self.iter_eigenvalues[-1] -
                                      self.iter_eigenvalues[-2])
                self.iter_precision.append(prec)
                if prec < self.eps:
                    break
        self.eigenvalues = sorted(self.iter_eigenvalues[-1], reverse=True)  # 最终特征值
        return self.eigenvalues

    @staticmethod
    def _miu_displacement_wilkins_(orthogonal_mat):
        """
        威尔金斯平移方法，选择μ
        :return:
        """
        n = orthogonal_mat.shape[0]
        sub_mat = orthogonal_mat[n - 2:, n - 2:]
        t = sympy.Symbol("t")  # 符号变量
        chara_poly = sympy.det(sub_mat - t * sympy.eye(2))  # 特征多项式
        polynomial = sympy.Poly(chara_poly, t)
        c = polynomial.coeffs()
        delta = c[1] ** 2 - 4 * c[0] * c[2]
        if len(c) == 3 and delta >= 0:
            eig_1 = (-c[1] + math.sqrt(delta)) / 2 / c[0]
            eig_2 = (-c[1] - math.sqrt(delta)) / 2 / c[0]
            # 选择最接近A(n,n)的那个作为μ
            tmp1, tmp2 = eig_1 - orthogonal_mat[-1, -1], eig_2 - orthogonal_mat[-1, -1]
            miu = eig_1 if abs(tmp1) < abs(tmp2) else eig_2
        else:  # 两个特征值有一个为复数
            miu = orthogonal_mat[-1, -1]
        return float(miu)