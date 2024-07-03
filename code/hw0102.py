import numpy as np
from HeatDiffusionVariance import HDV02

# # Set model hyperparameters(time_axis, space_axis, K, A)
time_axis_ = np.arange(0, 10, 0.001)
space_axis_ = np.arange(0, 1, 0.01)
K_ = 0.01
#
hdv02 = HDV02(time_axis_, space_axis_, K_)
#
T_init_ = hdv02.init_T()
#
T_as_ = hdv02.run_analytical_solution()
T_ns_ = hdv02.run_numerical_solution()

# draw gif
filepath_ = '../fig/hw0102_hdv_1d_ftcs_ns.gif'
hdv02.draw_gif(T_ns_, 100, 100, 5, filepath_)

filepath_ = '../fig/hw0102_hdv_1d_ftcs_as.gif'
hdv02.draw_gif(T_as_, 100, 100, 5, filepath_)
