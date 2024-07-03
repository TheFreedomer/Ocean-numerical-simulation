import numpy as np
from ConvectiveDiffusion import CD01, CD02


# # Set model hyperparameters(time_axis, space_axis, K, A)
time_axis_ = np.arange(0, 50, 0.001)
time_axis_02 = np.arange(0, 100, 1.5)
space_axis_ = np.arange(0, 1, 0.01)
K_ = 0.01
U_ = 0.01
bc_ = 2     # 第二类边界条件

# #
cd01 = CD01(time_axis_, space_axis_, K_, U_)
T_ns_ = cd01.run_numerical_solution()
# draw gif
filepath_ = '../fig/hw0107_cd_1d_ftcs_ns.gif'
cd01.draw_gif(T_ns_, 200, 100, 20, filepath_)

# #
cd02 = CD02(time_axis_02, space_axis_, K_, U_, bc_)
T_ns_ = cd02.run_numerical_solution()
# draw gif
filepath_ = '../fig/hw0107_cd_1d_btcs_ns.gif'
cd02.draw_gif(T_ns_, 1, 100, 8, filepath_)
