import matplotlib.pyplot as plt
import imageio
import io
import numpy as np


class BNA01(object):
    """
    One-Dimensional Bugger's nonlinear advection equation(Difference format01)
    """

    def __init__(self, time_axis, space_axis):
        """
        Here you need to define some parameters necessary for the model to start correctly.

        --------------------------------------------------------------------------------------
        A sample follow >>>


        --------------------------------------------------------------------------------------

        :param time_axis: Time --> ndarray[float]
        :param space_axis: X --> ndarray[float]
        """
        # 空间轴
        self.time_axis = time_axis
        # 时间轴
        self.space_axis = space_axis
        # dt
        self.dt = self.time_axis[1] - self.time_axis[0]
        # dx
        self.dx = self.space_axis[1] - self.space_axis[0]
        # r
        self.r = self.dt / self.dx

    def init_U(self):
        """
        The condition of Temperature(t = 0)

        :return: The condition of Temperature --> ndarray[float]
        """
        U_init = np.sin(2 * np.pi * self.space_axis)
        return U_init

    def run_numerical_solution(self):
        """
        Running numerical solution by difference format 01

        :return: The numerical solution --> ndarray[float]
        """
        # vessel_target
        U = np.zeros((self.time_axis.size, self.space_axis.size))
        # initial condition
        U_pre = self.init_U()
        U_current = U_pre
        # U_current = np.zeros_like(U_pre)
        # update U
        U[0, :] = U_pre
        U[1, :] = U_current
        # count
        for i in range(2, self.time_axis.size - 2):
            # vessel
            U_next = np.zeros_like(U_current)
            # idx_inner
            idx_inner = np.array(range(1, U_next.size - 1))
            # update middle
            U_next[idx_inner] = U_pre[idx_inner] - self.r / 4 * (
                    (U_current[idx_inner + 1] + U_current[idx_inner]) ** 2 -
                    (U_current[idx_inner - 1] + U_current[idx_inner]) ** 2)
            # update endpoint
            U_next[0] = U_next[-2]
            U_next[-1] = U_next[1]
            # update U
            U[i, :] = U_next
            # update U_current
            U_pre = U_current
            U_current = U_next
        return U

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
            plt.ylabel('U(m/s)', fontsize=fontsize + 2)
            # save to buffer
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=dpi)
            plt.close()
            # reset point
            buffer.seek(0)
            # imageio read fig
            frames.append(imageio.v2.imread(buffer))
        imageio.mimsave(filepath, frames, fps=fps)


class BNA02(BNA01):
    """
    Same as BAN01 but with different initial condition
    """

    def __init__(self, time_axis, space_axis):
        """

        :param time_axis:
        :param space_axis:
        """
        super().__init__(time_axis, space_axis)

    def init_U(self):
        """
        The condition of Temperature(t = 0)

        :return: The condition of Temperature --> ndarray[float]
        """
        U_init = 1.5 + np.sin(2 * np.pi * self.space_axis)
        return U_init


class BNA03(BNA01):
    """
    One-Dimensional Bugger's nonlinear advection equation(Difference format02) with initial condition 01
    """

    def __init__(self, time_axis, space_axis):
        """

        :param time_axis:
        :param space_axis:
        """
        super().__init__(time_axis, space_axis)

    def run_numerical_solution(self):
        """
        Run by Difference format02 with condition 01
        :return:
        """
        # vessel_target
        U = np.zeros((self.time_axis.size, self.space_axis.size))
        # initial condition
        U_pre = self.init_U()
        U_current = U_pre
        # U_current = np.zeros_like(U_pre)
        # update U
        U[0, :] = U_pre
        U[1, :] = U_current
        for i in range(2, self.time_axis.size - 2):
            # vessel
            U_next = np.zeros_like(U_current)
            # U_
            U_ = 0.5 * (U_pre + U_current)
            # idx_inner
            idx_inner = np.array(range(1, U_next.size - 1))
            # update middle
            # U_next[idx_inner] = U_pre[idx_inner] - 1/3 * self.r * (
            #     U_[idx_inner] * (U_[idx_inner + 1] + U_[idx_inner - 1]) +
            #     U_[idx_inner + 1] ** 2 - U_[idx_inner - 1] ** 2)
            U_next[idx_inner] = U_pre[idx_inner] - self.dt * (U_[idx_inner] + U_[idx_inner + 1] - U_[idx_inner - 1]) \
                                + (U_[idx_inner + 1] ** 2 - U_[idx_inner - 1] ** 2) / (3 * self.dx)
            # update endpoint
            U_next[0] = U_next[-2]
            U_next[-1] = U_next[1]
            # update U
            U[i, :] = U_next
            # update U_current U_pre
            U_pre = U_current
            U_current = U_next
        return U


class BNA04(BNA03):
    """
    Same as BNA03 with initial condition 02
    """

    def __init__(self, time_axis, space_axis):
        """

        :param time_axis:
        :param space_axis:
        """
        super().__init__(time_axis, space_axis)

    def init_U(self):
        """
        The condition of Temperature(t = 0)

        :return: The condition of Temperature --> ndarray[float]
        """
        U_init = 1.5 + np.sin(2 * np.pi * self.space_axis)
        return U_init


if __name__ == '__main__':
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

    # # #
    # bna03 = BNA03(time_axis_, space_axis_)
    # T_ns_ = bna03.run_numerical_solution()
    # # draw gif
    # filepath_ = '../fig/hw0201_bna_1d_format02_condition01_ns.gif'
    # bna03.draw_gif(T_ns_, 50, 100, 10, filepath_)

    # # #
    # bna04 = BNA04(time_axis_, space_axis_)
    # T_ns_ = bna04.run_numerical_solution()
    # # draw gif
    # filepath_ = '../fig/hw0201_bna_1d_format02_condition02_ns.gif'
    # bna04.draw_gif(T_ns_, 50, 100, 10, filepath_)
    pass
