import numpy as np
from Advection import Advection01, Advection02

time_axis_ = np.arange(0, 100, 0.1)
space_axis_ = np.arange(0, 1, 0.01)
U_ = 0.01

# instantiation
adv01 = Advection01(time_axis_, space_axis_, U_)
# numerical solution
T_ns_ = adv01.run_numerical_solution()
# draw gif
filepath_ = '../fig/hw0106_adv_1d_ctcs_ns.gif'
adv01.draw_gif(T_ns_, 5, 100, 50, filepath_)

# instantiation
adv02 = Advection02(time_axis_, space_axis_, U_)
# numerical solution
T_ns_ = adv02.run_numerical_solution()
# draw gif
filepath_ = '../fig/hw0106_adv_1d_ftbs_ns.gif'
adv02.draw_gif(T_ns_, 5, 100, 50, filepath_)
