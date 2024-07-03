import matplotlib.pyplot as plt
import imageio
import io
import numpy as np


class CD01(object):
    """
    One-Dimensional Convective Diffusion(FTCS, Forward Time Central Space)
    """

    def __init__(self, time_axis, space_axis, K, U):
        """
        Here you need to define some parameters necessary for the model to start correctly,
        you need to note that the stability coefficient self.lamda must be < 1,
        otherwise the numerical pattern will diverge and the program will raise ValueError.

        --------------------------------------------------------------------------------------
        A sample follow >>>


        --------------------------------------------------------------------------------------

        :param time_axis: Time --> ndarray[float]
        :param space_axis: X --> ndarray[float]
        :param K: Diffusion coefficient --> [float]
        :param U: Advection velocity --> [float]
        """
        # 空间轴
        self.time_axis = time_axis
        # 时间轴
        self.space_axis = space_axis
        # 扩散系数
        self.K = K
        # 速度大小
        self.U = U
        # dt
        self.dt = self.time_axis[1] - self.time_axis[0]
        # dx
        self.dx = self.space_axis[1] - self.space_axis[0]
        # miu
        self.miu = K * self.dt / self.dx ** 2
        # lamda
        self.lamda = U * self.dt / self.dx

        # print
        print('MIU = ', '\t', self.miu)
        print('Lamda = ', '\t', self.lamda)

    def init_T(self):
        """
        The condition of Temperature(t = 0)

        :return: The condition of Temperature --> ndarray[float]
        """
        T_init = np.exp(-(self.space_axis - self.space_axis[-1] / 2) ** 2 / (0.1 * self.space_axis[-1]) ** 2)

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
            T_next[idx_inner] = (1 - 2 * self.miu) * T_current[idx_inner] + \
                                (self.miu - self.lamda / 2) * T_current[idx_inner + 1] + \
                                (self.miu + self.lamda / 2) * T_current[idx_inner - 1]
            # update endpoint
            T_next[0] = T_next[1]
            T_next[-1] = T_next[-2]
            # update T
            T[i, :] = T_next
            # update T_current
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


class CD02(CD01):
    """
    Same as CD01 but with BTCS(Backward Time Central Space)
    """
    def __init__(self, time_axis, space_axis, K, U, bc=2):
        """
        Same as CD01
        :param time_axis:
        :param space_axis:
        :param K:
        :param U:
        :param bc: Boundary condition --> [int]
        """
        super().__init__(time_axis, space_axis, K, U)
        self.bc = bc

    def create_D(self):
        """
        Construct the difference matrix D
        :return: difference matrix D --> ndarray[sequence]
        """
        # num of space grid
        num = self.space_axis.size
        # coefficient
        coefficient = np.array([self.miu + self.lamda / 2,
                                -(1 + 2 * self.miu),
                                self.miu - self.lamda / 2])
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
    cd02.draw_gif(T_ns_, 1, 100, 5, filepath_)
