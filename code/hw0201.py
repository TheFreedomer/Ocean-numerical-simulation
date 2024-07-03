import numpy as np
from BuggerNonlinearAdvection import BNA01, BNA02

# # Set model hyperparameters(time_axis, space_axis, K, A)
time_axis_ = np.arange(0, 3, 0.001)
time_axis_02 = np.arange(0, 5, 0.001)
space_axis_ = np.arange(0, 1, 0.01)

# #
bna01 = BNA01(time_axis_, space_axis_)
T_ns_ = bna01.run_numerical_solution()
# draw gif
filepath_ = '../fig/hw0201_bna_1d_format01_condition01_ns.gif'
bna01.draw_gif(T_ns_, 50, 100, 10, filepath_)

# #
bna02 = BNA02(time_axis_02, space_axis_)
T_ns_ = bna02.run_numerical_solution()
# draw gif
filepath_ = '../fig/hw0201_bna_1d_format01_condition02_ns.gif'
bna02.draw_gif(T_ns_, 50, 100, 10, filepath_)
