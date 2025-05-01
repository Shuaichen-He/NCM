# -*- coding: UTF-8 -*-
"""
@file_name: test_guass_legendre_3d_int_13.py
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
from numerical_integration_04.guass_legendre_3d_integration import GaussLegendreTripleIntegration
from util_font import *

int_fun = lambda x, y, z: 4 * x * z * np.exp(-x ** 2 * y - z ** 2)

zeros_num = np.arange(10, 26, 1)
int_res = np.zeros(len(zeros_num))
for i, n in enumerate(zeros_num):
    gl3d = GaussLegendreTripleIntegration(int_fun, [0, 1], [0, np.pi], [0, np.pi], zeros_num=[n, n, n])
    gl3d.cal_3d_int()
    int_res[i] = np.abs(1.7327622230312205 - gl3d.int_value)
    print(n, gl3d.int_value, 1.7327622230312205 - gl3d.int_value)

plt.figure(figsize=(7, 5))
plt.semilogy(zeros_num, int_res, "o-", lw=1.5)
idx = np.argmin(int_res)
plt.semilogy(zeros_num[idx], int_res[idx], "D", label="$n=%d,\ \epsilon=%.5e$" % (zeros_num[idx], int_res[idx]))
plt.xlabel("零点数$n$", fontdict={"fontsize": 18})
plt.ylabel(r"$\epsilon=\vert I - I^* \vert$", fontdict={"fontsize": 18})
plt.title("高斯—勒让德三重积分收敛性", fontdict={"fontsize": 18})
plt.tick_params(labelsize=16)  # 刻度字体大小16
plt.legend(frameon=False, fontsize=18)
plt.grid(ls=":")
plt.show()

glti = GaussLegendreTripleIntegration(int_fun, [0, 1], [0, np.pi], [0, np.pi], zeros_num=[11, 12, 15])
res = glti.cal_3d_int()
print("积分值：%.15f， 精度：%.15e" % (res, 1.7327622230312205 - res))
