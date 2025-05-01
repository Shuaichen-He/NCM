# -*- coding: UTF-8 -*-
"""
@file_name: exp3.py
@time: 2023-02-10
@IDE: PyCharm  Python: 3.9.7
@copyright: http://maths.xynu.edu.cn
"""
import numpy as np
import matplotlib.pyplot as plt
from function_approximation_03.least_square_curve_fitting import LeastSquarePolynomialCurveFitting
from function_approximation_03.orthogonal_polynomial_least_squares_fitting import OrthogonalPolynomialLSFitting

# 实验数据
x = np.linspace(-1, 5, 16)
y = np.array([-2.11802123, -0.70930496, 2.12377848, 0.85944498, 0.98580008, -0.51909279, 0.4820484,
              -2.32350214, -2.60275523, -4.15452315, -3.27630398, -3.65130248, -1.56691817,
              0.77433117, 2.37330911, 8.89881447])

# 多项式最小二乘拟合
orders = [2, 3, 5, 8]
xi = np.linspace(min(x), max(x), 100, endpoint=True)
plt.figure(figsize=(14, 5))
plt.subplot(121)
line_style = ["-", "--", "-.", ":"]
for i, order in enumerate(orders):
    lspcf = LeastSquarePolynomialCurveFitting(x, y, k=order)
    lspcf.fit_ls_curve()
    print("order = %d：" % order, lspcf.poly_coefficient, lspcf.mse)
    yi = lspcf.predict_x0(xi)
    plt.plot(xi, yi, line_style[i], lw=2, label="$k = %d, \ MSE = %.2e$" % (order, lspcf.mse))
plt.plot(x, y, "ko", label="$(x_k, y_k)$")
plt.legend(frameon=False, fontsize=18)
plt.grid(ls=":")
plt.tick_params(labelsize=16)  # 刻度字体大小16
plt.xlabel("$x$", fontdict={"fontsize": 18})
plt.ylabel("$p(x)$", fontdict={"fontsize": 18})
plt.title("不同阶次的最小二乘多项式曲线拟合", fontdict={"fontsize": 18})

print("=" * 80)

# 正交多项式最小二乘拟合
plt.subplot(122)
for i, order in enumerate(orders):
    oplsf = OrthogonalPolynomialLSFitting(x, y, k=order)
    oplsf.fit_orthogonal_poly()
    print("order = %d：" % order, oplsf.poly_coefficient, oplsf.mse)
    yi = oplsf.predict_x0(xi)
    plt.plot(xi, yi, line_style[i], lw=2, label="$k = %d, \ MSE = %.2e$" % (order, oplsf.mse))
plt.plot(x, y, "ko", label="$(x_k, y_k)$")
plt.legend(frameon=False, fontsize=18)
plt.grid(ls=":")
plt.tick_params(labelsize=16)  # 刻度字体大小16
plt.xlabel("$x$", fontdict={"fontsize": 18})
plt.ylabel("$p(x)$", fontdict={"fontsize": 18})
plt.title("不同阶次的正交多项式最小二乘曲线拟合", fontdict={"fontsize": 18})
plt.show()