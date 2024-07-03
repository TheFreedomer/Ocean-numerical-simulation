import matplotlib.pyplot as plt
import imageio
import io
import numpy as np


class HDV01(object):
    """
    One-Dimensional Heat Diffusion Variance(FTCS, Forward Time Central Space)
    """

    def __init__(self, time_axis, space_axis, K, A):
        """
        Here you need to define some parameters necessary for the model to start correctly,
        you need to note that the stability coefficient self.miu must be <=0.5,
        otherwise the numerical pattern will diverge and the program will raise ValueError.

        --------------------------------------------------------------------------------------
        A sample follow >>>

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
        --------------------------------------------------------------------------------------

        :param time_axis: Time --> ndarray[float]
        :param space_axis: X --> ndarray[float]
        :param K: Diffusion coefficient --> [float]
        :param A: Initial state correlation coefficient --> [float]
        """
        # 空间轴
        self.time_axis = time_axis
        # 时间轴
        self.space_axis = space_axis
        # 扩散系数
        self.K = K
        # dt
        self.dt = self.time_axis[1] - self.time_axis[0]
        # dx
        self.dx = self.space_axis[1] - self.space_axis[0]
        # 稳定参数
        self.miu = K * self.dt / self.dx ** 2
        #
        if self.miu > 0.5:
            raise ValueError('The lamda must <= 0.5')

        # print
        print('MIU = ', '\t', self.miu)

        # 初始场相关参数
        self.A = A
        #
        self.lambda_ = np.pi / self.space_axis[-1]

    def init_T(self):
        """
        The condition of Temperature(t = 0)

        :return: The condition of Temperature --> ndarray[float]
        """
        T_init = self.A * np.cos(self.lambda_ * self.space_axis)
        return T_init

    def run_numerical_solution(self):
        """
        Running numerical solution by forward-Time-Central-Space way

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
            T_next[idx_inner] = self.miu * (T_current[idx_inner + 1] + T_current[idx_inner - 1]) \
                                + (1 - 2 * self.miu) * T_current[idx_inner]
            # update endpoint
            T_next[0] = T_next[1]
            T_next[-1] = T_next[-2]
            # update T
            T[i, :] = T_next
            # update T_current
            T_current = T_next
        return T

    def f_analytical_solution(self, t, x):
        """
        The function of analytical solution

        :param t: Time --> [float]
        :param x: Position --> [float]|ndarray[float]
        :return: The analytical solution of t --> ndarray[float]
        """

        T_as = self.A * np.exp(-self.K * (self.lambda_ ** 2) * t) * np.cos(self.lambda_ * x)
        return T_as

    def run_analytical_solution(self):
        """
        Calculate analytical solution

        :return:All the analytical solution --> ndarray[float]
        """
        # vessel_target
        T = np.zeros((self.time_axis.size, self.space_axis.size))
        for i, t in enumerate(self.time_axis):
            T[i, :] = self.f_analytical_solution(t, self.space_axis)
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


class HDV02(HDV01):
    """
    One-Dimensional Heat Diffusion Variance(FTCS, Forward Time Central Space).
    Compared with HDV01, the initial temperature field and analytical solution in this class are different,
    and the corresponding method is rewritten later.
    """
    def __init__(self, time_axis, space_axis, K):
        """
        Same with HDV01
        """
        # inherited superclass
        super().__init__(time_axis, space_axis, K, A=0)

    def init_T(self):
        """
        The condition of Temperature(t = 0)

        :return: The condition of Temperature --> ndarray[float]
        """
        T_init = np.exp(-(self.space_axis - self.space_axis[-1] / 2) ** 2 / (0.1 * self.space_axis[-1]) ** 2)
        return T_init

    def run_analytical_solution(self):
        """
        Calculate analytical solution

        :return:All the analytical solution --> ndarray[float]
        """
        # vessel
        T = np.zeros((self.time_axis.size, self.space_axis.size), dtype=complex)
        # info
        num = self.space_axis.size
        L = self.space_axis[-1]
        # an
        an = 2 / num * np.fft.fft(self.init_T())
        an = an.reshape(-1, 1)
        # kn
        kn_01 = np.arange(0, num//2 + 1)
        kn_02 = -np.arange(1, num//2)[::-1]
        kn = np.concatenate((kn_01, kn_02)).reshape((-1, 1)) * 2 * np.pi / L
        # exp(i * kn * x)
        term02 = np.exp(1j * (kn @ self.space_axis.reshape((1, -1))))
        # run
        for i, t in enumerate(self.time_axis):
            term01 = (an * np.exp(-self.K * kn ** 2 * t)).T
            T[i, :] = term01 @ term02
        return np.real(T)


class HDV03(HDV02):
    """
    One-Dimensional Heat Diffusion Variance(BTCS, Backward Time Central Space).
    The difference format in this class is different compared to HDV02
    """
    def __init__(self, time_axis, space_axis, K, bc):
        """

        :param time_axis:
        :param space_axis:
        :param K:
        :param bc: The boundary condition must be 1 or 2 --> int
        """
        # inherited superclass
        super().__init__(time_axis, space_axis, K)
        self.bc = bc

    def create_D(self):
        """
        Construct the difference matrix D
        :return: difference matrix D --> ndarray[sequence]
        """
        # num of space grid
        num = self.space_axis.size
        # coefficient
        coefficient = np.array([self.miu, -(1 + 2 * self.miu), self.miu])
        # init
        D = np.zeros((num - 2, num - 2))
        # update boundary region
        D[0, 0: 2] = coefficient[1:]
        D[-1, -2:] = coefficient[:-1]
        # update interior zone
        for i in range(1, num - 2 - 1):
            D[i, i-1: i+2] = coefficient
        # boundary conditions
        if self.bc == 1:
            pass
        elif self.bc == 2:
            D[0, 0] = -(1 + self.miu)
            D[-1, -1] = -(1 + self.miu)
        else:
            raise ValueError('The boundary condition must be 1 or 2 --> int')

        return D

    def run_numerical_solution(self):
        """
        Run by BTCS
        :return:The numerical solution --> ndarray[float]
        """
        # vessel_target
        T = np.zeros((self.time_axis.size, self.space_axis.size))
        # initial condition
        T_current = self.init_T()
        # update T
        T[0, :] = T_current
        # D
        D = self.create_D()
        for i in range(1, self.time_axis.size - 1):
            # vessel
            T_next = np.zeros_like(T_current)
            # idx_inner
            idx_inner = np.array(range(1, T_next.size - 1))
            # D_inv
            D_inv = np.linalg.inv(D)
            # T_current
            T_next[idx_inner] = (-D_inv @ T_current.reshape(-1, 1)[idx_inner, :]).reshape((-1))
            # boundary condition
            if self.bc == 1:
                T_next[0] = 0
                T_next[-1] = 0
            elif self.bc == 2:
                T_next[0] = T_next[1]
                T_next[-1] = T_next[-2]
            else:
                raise ValueError('The boundary condition must be 1 or 2 --> int')
            # update T
            T[i, :] = T_next
            # update T_current
            T_current = T_next

        return T


if __name__ == '__main__':
    # # Set model hyperparameters(time_axis, space_axis, K, A)
    time_axis_ = np.arange(0, 50, 0.001)
    space_axis_ = np.arange(0, 1, 0.01)
    K_ = 0.01
    A_ = 10
    bc_ = 2

    # # HDV03
    hdv03 = HDV03(time_axis_, space_axis_, K_, bc_)

    T_init_ = hdv03.init_T()
    # T_ns_ = hdv03.run_numerical_solution()
    D_ = hdv03.create_D()
    T_ns_ = hdv03.run_numerical_solution()

    # draw gif
    filepath_ = '../fig/hw0104_hdv_1d_btcs_ns.gif'
    hdv03.draw_gif(T_ns_, 100, 100, 10, filepath_)
    #
    # filepath_ = '../fig/hw0101_hdv_1d_ftcs_as.gif'
    # adv01.draw_gif(T_as_, 500, 100, 10, filepath_)
