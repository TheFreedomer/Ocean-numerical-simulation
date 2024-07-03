import numpy as np
from HeatDiffusionVariance import HDV02, HDV03
import time


# # Set model hyperparameters(time_axis, space_axis, K, A)
time_axis_ = np.arange(0, 20, 0.001)
time_axis_02 = np.arange(0, 20, 0.002)  # 2*delta_t
space_axis_ = np.arange(0, 1, 0.01)
K_ = 0.01
A_ = 10
bc_ = 2

# # # experiment01
print('experiment01 >>> ')
# # HDV2(FTCS)
hdv02 = HDV02(time_axis_, space_axis_, K_)

time_s = time.time()
T_ns_ = hdv02.run_numerical_solution()
time_e = time.time()
print('FTCS use ', time_e - time_s, ' s')

# draw gif
filepath_ = '../fig/hw0104_hdv_1d_ftcs_ns.gif'
hdv02.draw_gif(T_ns_, 100, 100, 10, filepath_)

# # HDV2(BTCS)
hdv03 = HDV03(time_axis_, space_axis_, K_, bc_)

time_s = time.time()
T_ns_ = hdv03.run_numerical_solution()
time_e = time.time()
print('BTCS use ', time_e - time_s, ' s')

# draw gif
filepath_ = '../fig/hw0104_hdv_1d_btcs_ns.gif'
hdv03.draw_gif(T_ns_, 100, 100, 10, filepath_)

# # # experiment02
print('experiment02 >>> ')

# # HDV2(FTCS)
hdv02 = HDV02(time_axis_02, space_axis_, K_)

time_s = time.time()
T_ns_ = hdv02.run_numerical_solution()
time_e = time.time()
print('FTCS use ', time_e - time_s, ' s')

# draw gif
filepath_ = '../fig/hw0104_hdv_1d_ftcs_ns_2delta_t.gif'
hdv02.draw_gif(T_ns_, 100, 100, 10, filepath_)

# # HDV2(BTCS)
hdv03 = HDV03(time_axis_02, space_axis_, K_, bc_)

time_s = time.time()
T_ns_ = hdv03.run_numerical_solution()
time_e = time.time()
print('BTCS use ', time_e - time_s, ' s')

# draw gif
filepath_ = '../fig/hw0104_hdv_1d_btcs_ns_2delta_t.gif'
hdv03.draw_gif(T_ns_, 100, 100, 10, filepath_)
