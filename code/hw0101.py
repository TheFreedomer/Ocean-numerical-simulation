from HeatDiffusionVariance import HDV01
import numpy as np


# # Set model hyperparameters(time_axis, space_axis, K, A)
time_axis_ = np.arange(0, 50, 0.001)
space_axis_ = np.arange(0, 1, 0.01)
K_ = 0.01
A_ = 10
hdv = HDV01(time_axis_, space_axis_, K_, A_)

# T_init_ = adv01.init_T()
T_ns_ = hdv.run_numerical_solution()
T_as_ = hdv.run_analytical_solution()

# draw gif
filepath_ = '../fig/hw0101_hdv_1d_ftcs_ns.gif'
hdv.draw_gif(T_ns_, 500, 100, 10, filepath_)

filepath_ = '../fig/hw0101_hdv_1d_ftcs_as.gif'
hdv.draw_gif(T_as_, 500, 100, 10, filepath_)
