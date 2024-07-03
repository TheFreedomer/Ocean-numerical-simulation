import math
from scipy import integrate


def get_Sn_recursion(n):
    """
    递归算法计算
    :param n: [int]
    :return:
    """
    if n == 0:
        return math.log(6) - math.log(5)
    else:
        return 1 / n - 5 * get_Sn_recursion(n - 1)


def get_Sn_direct(n):
    """
    直接计算
    :param n:
    :return:
    """
    # 总和
    Sn = 0
    # 初始化S0
    Sn_1 = round((math.log(6) - math.log(5)), 3)    # 保留3位小数
    for n_current in range(1, n + 1):
        # 计算当前n_current对应结果
        result_current = round((1 / n_current - 5 * Sn_1), 3)   # 保留3位小数
        # 更新Sn
        Sn += result_current
        # 更新Sn_1
        Sn_1 = Sn
    return Sn


def f_target(x, n):
    """

    :param x:
    :param n:
    :return:
    """
    return x ** n / (x + 5)


def get_Sn_integrate(n):
    """

    :param n:
    :return:
    """
    Sn, *_ = integrate.quad(f_target, 0, 1, args=n)
    return Sn


if __name__ == '__main__':
    n_ = 8
    print('递归算法计算结果(双精度浮点数计算): ', round(get_Sn_recursion(n_), 3))
    print('直接数值解求和计算结果(保留3位小数): ', round(get_Sn_direct(n_), 3))
    print('直接积分计算(双精度浮点数计算): ', round(get_Sn_integrate(n_), 3))
