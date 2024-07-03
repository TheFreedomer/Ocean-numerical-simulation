import numpy as np
from Advection import Advection01


#
time_axis_ = np.arange(0, 100, 0.001)
space_axis_ = np.arange(0, 1, 0.01)
U_ = 0.01
# instantiation
adv01 = Advection01(time_axis_, space_axis_, U_)
# numerical solution
T_ns_ = adv01.run_numerical_solution()
# draw gif
filepath_ = '../fig/hw0105_adv_1d_ctcs_ns.gif'
adv01.draw_gif(T_ns_, 500, 100, 50, filepath_)
