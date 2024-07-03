import matplotlib.pyplot as plt
import imageio
import io
import numpy as np


class Advection01(object):
    """
    One-Dimensional advection motion(CTCS(Leap Frogs), Central Time Central Space)
    """

    def __init__(self, time_axis, space_axis, U):
        """
        Here you need to define some parameters necessary for the model to start correctly,
        you need to note that the stability coefficient self.lamda must be < 1,
        otherwise the numerical pattern will diverge and the program will raise ValueError.

        --------------------------------------------------------------------------------------
        A sample follow >>>

        time_axis_ = np.arange(0, 200, 0.001)
        space_axis_ = np.arange(0, 1, 0.01)
        U_ = 0.1
        adv01 = Advection01(time_axis_, space_axis_, U_)
        # numerical solution
        T_ns_ = adv01.run_numerical_solution()
        # draw gif
        filepath_ = '../fig/hw0105_adv_1d_ctcs_ns.gif'
        adv01.draw_gif(T_ns_, 500, 100, 10, filepath_)
        --------------------------------------------------------------------------------------

        :param time_axis: Time --> ndarray[float]
        :param space_axis: X --> ndarray[float]
        :param U: Advection velocity --> [float]
        """
        # 空间轴
        self.time_axis = time_axis
        # 时间轴
        self.space_axis = space_axis
        # 流速
        self.U = U
        # dt
        self.dt = self.time_axis[1] - self.time_axis[0]
        # dx
        self.dx = self.space_axis[1] - self.space_axis[0]
        # 稳定参数
        self.lamda = U * self.dt / self.dx
        # 检查稳定性
        print('LAMDA = ', '\t', self.lamda)
        if self.lamda >= 1:
            raise ValueError('The lamda must < 1')

    def init_T(self):
        """
        The condition of Temperature(t = 0)

        :return: The condition of Temperature --> ndarray[float]
        """
        T_init = np.exp(-(self.space_axis - self.space_axis[-1] / 2) ** 2 / (0.1 * self.space_axis[-1]) ** 2)

        return T_init

    def run_numerical_solution(self):
        """
        Running numerical solution by Central-Time-Central-Space way

        :return: The numerical solution --> ndarray[float]
        """
        # vessel_target
        T = np.zeros((self.time_axis.size, self.space_axis.size))
        # initial condition
        T_pre = self.init_T()
        T_current = T_pre
        # update T
        T[0, :] = T_pre
        T[1, :] = T_pre
        # count
        for i in range(2, self.time_axis.size - 2):
            # vessel
            T_next = np.zeros_like(T_pre)
            # idx_inner
            idx_inner = np.array(range(1, T_next.size - 1))
            # update middle
            T_next[idx_inner] = T_pre[idx_inner] - self.lamda * (T_current[idx_inner + 1] - T_current[idx_inner - 1])
            # update endpoint
            T_next[0] = T_next[-2]
            T_next[-1] = T_next[1]
            # update T
            T[i, :] = T_next
            # update T_pre
            T_pre = T_current
            T_current = T_next
        return T

    def draw_gif(self, data, time_interval, dpi, fps, filepath):
        """
        Draw GIF based on calculation result
        :param data: The analytical solution or the numerical solution --> ndarray[float]
        :param time_interval: Interval how many time steps to add a frame --> [int]
        :param dpi: Dots Per Inch --> [int]
        :param fps: Nums of figure per second --> [int]
        :param filepath: Output path of file
        :return:
        """
        # frame sequence
        frames = []
        # size
        T_s, X_s = data.shape
        # info
        v_min, v_max = np.min(data), np.max(data)
        fontsize = 10
        for time_step in range(0, T_s, time_interval):
            # data of time_step
            data_step = data[time_step, :]
            # get_one_frame
            plt.figure(figsize=(9, 6))
            plt.plot(self.space_axis, data_step)
            plt.ylim(v_min, v_max)
            plt.xticks(fontsize=fontsize)
            plt.yticks(fontsize=fontsize)
            plt.title('t = ' + '{:.2f}'.format(self.time_axis[time_step]) + ' s', fontsize=fontsize + 4)
            plt.xlabel('X', fontsize=fontsize + 2)
            plt.ylabel('Temperature(℃)', fontsize=fontsize + 2)
            # save to buffer
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=dpi)
            plt.close()
            # reset point
            buffer.seek(0)
            # imageio read fig
            frames.append(imageio.v2.imread(buffer))
        imageio.mimsave(filepath, frames, fps=fps)


class Advection02(Advection01):
    """
    One-Dimensional advection motion(Upwind format, Forward Time Backward Space)
    """

    def __init__(self, time_axis, space_axis, U):
        """
        Same with Advection01
        :param time_axis:
        :param space_axis:
        :param U:
        """
        super().__init__(time_axis, space_axis, U)

    def run_numerical_solution(self):
        """
        Running numerical solution by Forward-Time-Backward-Space way

        :return: The numerical solution --> ndarray[float]
        """
        # vessel_target
        T = np.zeros((self.time_axis.size, self.space_axis.size))
        # initial condition
        T_current = self.init_T()
        # update T
        T[0, :] = T_current
        # count
        for i in range(1, self.time_axis.size - 1):
            # vessel
            T_next = np.zeros_like(T_current)
            # idx_inner
            idx_inner = np.array(range(1, T_next.size - 1))
            # update middle
            T_next[idx_inner] = T_current[idx_inner] - self.lamda * (T_current[idx_inner + 1] - T_current[idx_inner - 1])
            # update endpoint
            T_next[0] = T_next[-2]
            T_next[-1] = T_next[1]
            # update T
            T[i, :] = T_next
            # update T_current
            T_current = T_next
        return T


if __name__ == '__main__':
    time_axis_ = np.arange(0, 50, 0.001)
    space_axis_ = np.arange(0, 1, 0.01)
    U_ = 0.1

    # instantiation
    adv01 = Advection01(time_axis_, space_axis_, U_)
    # numerical solution
    T_ns_ = adv01.run_numerical_solution()
    # draw gif
    filepath_ = '../fig/hw0105_adv_1d_ctcs_ns.gif'
    adv01.draw_gif(T_ns_, 500, 100, 10, filepath_)

    # instantiation
    adv02 = Advection02(time_axis_, space_axis_, U_)
    # numerical solution
    T_ns_ = adv02.run_numerical_solution()
    # draw gif
    filepath_ = '../fig/hw0106_adv_1d_ftbs_ns.gif'
    adv02.draw_gif(T_ns_, 500, 100, 10, filepath_)
