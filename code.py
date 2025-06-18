import numpy as np
import os
import matplotlib.pyplot as plt
from enum import Enum
from scipy.optimize import fsolve
from PIL import Image
from scipy.io import savemat
import matplotlib.animation as animation
#%% 一维可压缩流动数值模拟
#%% 研究案例：Sod激波管问题
#%% 类型：近似黎曼解
#%% 求解一维欧拉方程
#%% 使用可压缩流求解器
#%% 涉及通量分裂(FVS/FDS) + 高分辨率迎风格式(UPW)/激波捕捉格式(SCS)
#%% 版本：2.0
#%% 日期：2021/12/21

#%% 作业：计算流体动力学基础
#%% 姓名：冯正昊
#%% 学院：工程学院
#%% 学号：2101112008

#%% 程序初始化
def Cal_Minmod(a, b):
    """
    Minmod限制器函数

    参数:
    a, b -- 数组或标量值

    返回:
    minmod值 - 与输入相同的形状
    """
    # 如果输入是标量转为数组
    if np.isscalar(a) and np.isscalar(b):
        a = np.array([a])
        b = np.array([b])

    # 确保输入为 NumPy 数组
    a_arr = np.asarray(a)
    b_arr = np.asarray(b)

    # 初始化结果数组
    Q = np.zeros_like(a_arr)

    # 遍历所有元素
    for i in range(a_arr.size):
        # 展平多维数组
        a_i = a_arr.flat[i]
        b_i = b_arr.flat[i]

        if (a_i * b_i) > 0:
            # a和b同号，返回绝对值较小的那个
            if abs(a_i) > abs(b_i):
                Q.flat[i] = b_i
            else:
                Q.flat[i] = a_i
        else:
            # a和b异号，返回0
            Q.flat[i] = 0

    # 如果原始输入是标量，返回标量
    if np.isscalar(a) and np.isscalar(b):
        return Q.item()

    return Q
def Diff_Cons_Common(N, dx, F_p, F_n, flag_spa_typ, flag_upw_typ, flag_scs_typ):
    """
    通用通量差分计算函数(守恒形式)

    参数:
    N -- 网格点数
    dx -- 网格尺寸
    F_p -- 正通量分量 (N×3)
    F_n -- 负通量分量 (N×3)
    flag_spa_typ -- 空间离散类型标志
    flag_upw_typ -- 迎风格式标志
    flag_scs_typ -- 激波捕捉格式标志

    返回:
    xs_new, xt_new, Fh_p, Fh_n, Fx, Fx_p, Fx_n -- 索引和通量导数
    """
    # 初始化输出数组
    Fx = np.zeros((N, 3))  # 总通量导数
    Fx_p = np.zeros((N, 3))  # 正通量导数
    Fx_n = np.zeros((N, 3))  # 负通量导数
    Fh_p = np.zeros((N, 3))  # 半网格点正通量 (j+1/2)
    Fh_n = np.zeros((N, 3))  # 半网格点负通量 (j+1/2)
    Fh = np.zeros((N, 3))  # 半网格点总通量 (j+1/2)

    if flag_spa_typ == 1:
        # 1 - 一般迎风/紧致格式

        if flag_upw_typ == 1:
            # 1 - 一阶迎风格式 (2点)
            ks = -1  # 相对于j的起始索引
            kt = 0
            kn = kt - ks + 1  # 系数数量
            kp = np.array([ks, kt])  # [-1, 0]

            a = np.zeros(2)
            a[0] = -1
            a[1] = 1

            # 计算b系数
            b = np.zeros(kn - 1)
            b[0] = -a[0]

        elif flag_upw_typ == 2:
            # 2 - 二阶迎风格式 (3点)
            ks = -2
            kt = 0
            kn = kt - ks + 1  # 3
            kp = np.arange(ks, kt + 1)  # [-2, -1, 0]

            a = np.zeros(3)
            a[0] = 0.5
            a[1] = -2.0
            a[2] = 1.5

            # 计算b系数
            b = np.zeros(kn - 1)
            b[0] = -a[0]
            b[1] = b[0] - a[1]

        elif flag_upw_typ == 3:
            # 3 - 三阶迎风格式 (4点偏置)
            ks = -2
            kt = 1
            kn = kt - ks + 1  # 4
            kp = np.arange(ks, kt + 1)  # [-2, -1, 0, 1]

            a = np.zeros(4)
            a[0] = 1 / 6
            a[1] = -1.0
            a[2] = 0.5
            a[3] = 1 / 3

            # 计算b系数
            b = np.zeros(kn - 1)
            b[0] = -a[0]
            b[1] = b[0] - a[1]
            b[2] = b[1] - a[2]

        elif flag_upw_typ == 4:
            # 4 - 五阶迎风格式 (6点偏置)
            ks = -3
            kt = 2
            kn = kt - ks + 1  # 6
            kp = np.arange(ks, kt + 1)  # [-3, -2, -1, 0, 1, 2]

            a = np.zeros(6)
            a[0] = -2 / 60
            a[1] = 15 / 60
            a[2] = -60 / 60
            a[3] = 20 / 60
            a[4] = 30 / 60
            a[5] = -3 / 60

            # 计算b系数
            b = np.zeros(kn - 1)
            b[0] = -a[0]
            for k in range(1, kn - 1):
                b[k] = b[k - 1] - a[k]

        # [核心算法]
        # 根据系数计算通量
        xs = abs(kp[1])  # 起始索引
        xt = N - 1 - abs(kp[1])  # 结束索引

        # 计算半网格点通量
        for j in range(xs, xt):
            for k in range(len(b)):
                # F+ (j+1/2)
                Fh_p[j] += b[k] * F_p[j + kp[k + 1]]
                # F- (j-1/2)
                Fh_n[j] += b[k] * F_n[j - kp[k + 1]]

        # 更新索引范围
        xs_new = xs + 1
        xt_new = xt - 1

        # 计算通量导数
        for j in range(xs_new, xt_new):
            # 正通量导数
            Fx_p[j] = (Fh_p[j] - Fh_p[j - 1]) / dx
            # 负通量导数
            Fx_n[j] = (Fh_n[j + 1] - Fh_n[j]) / dx
            # 总通量导数
            Fx[j] = Fx_p[j] + Fx_n[j]

    elif flag_spa_typ == 2:
        # 2 - 特殊激波捕捉格式

        if flag_scs_typ == 1:
            # 1 - TVD格式 (Van Leer限制器)
            xs = 2  # 起始索引
            xt = N - 1 - xs  # 结束索引
            em = 1e-5  # 小量防止除零

            for j in range(xs, xt):
                # 计算正通量比例因子
                r_p_num = F_p[j] - F_p[j - 1]
                r_p_den = F_p[j + 1] - F_p[j] + em
                r_p = r_p_num / r_p_den

                # 计算负通量比例因子
                r_n_num = F_n[j + 2] - F_n[j + 1]
                r_n_den = F_n[j + 1] - F_n[j] + em
                r_n = r_n_num / r_n_den

                # Van Leer限制器
                Phi_p = (r_p + np.abs(r_p)) / (1 + r_p)
                Phi_n = (r_n + np.abs(r_n)) / (1 + r_n)

                # 计算半网格点通量
                Fh_p[j] = F_p[j] + 0.5 * Phi_p * (F_p[j + 1] - F_p[j])
                Fh_n[j] = F_n[j + 1] - 0.5 * Phi_n * (F_n[j + 1] - F_n[j])

            # 更新索引范围
            xs_new = xs + 1
            xt_new = xt

            # 计算通量导数
            for j in range(xs_new, xt_new):
                Fx_p[j] = (Fh_p[j] - Fh_p[j - 1]) / dx
                Fx_n[j] = (Fh_n[j] - Fh_n[j - 1]) / dx
                Fx[j] = Fx_p[j] + Fx_n[j]

        elif flag_scs_typ == 2:
            # 2 - NND格式 (无震荡、无自由参数耗散差分格式)
            xs = 2  # 起始索引
            xt = N - 1 - xs  # 结束索引

            for j in range(xs, xt):
                # 使用Minmod限制器计算半网格点通量
                mm_pos = Cal_Minmod(F_p[j] - F_p[j - 1], F_p[j + 1] - F_p[j])
                mm_neg = Cal_Minmod(F_n[j + 1] - F_n[j], F_n[j + 2] - F_n[j + 1])

                Fh_p[j] = F_p[j] + 0.5 * mm_pos
                Fh_n[j] = F_n[j + 1] - 0.5 * mm_neg
                Fh[j] = Fh_p[j] + Fh_n[j]  # 总通量

            # 更新索引范围
            xs_new = xs + 1
            xt_new = xt

            # 计算总通量导数
            for j in range(xs_new, xt_new):
                Fx[j] = (Fh[j] - Fh[j - 1]) / dx

        elif flag_scs_typ == 3:
            # 3 - WENO格式 (Jiang & Shu, 1996, 5阶)
            # 参数设置
            C = np.array([0.1, 0.6, 0.3])  # 理想权重
            p = 2  # 权重指数
            em = 1e-6  # 小量防止除零

            # 初始化
            xs = 3  # 起始索引
            xt = N - 1 - xs + 1  # 结束索引

            for j in range(xs, xt):
                # === 正通量分量 (F+) ===
                # 计算光滑指示器
                beta_p = np.zeros(3)
                beta_p[0] = (0.25 * (F_p[j - 2] - 4 * F_p[j - 1] + 3 * F_p[j]) ** 2 +
                             (13 / 12) * (F_p[j - 2] - 2 * F_p[j - 1] + F_p[j]) ** 2)
                beta_p[1] = (0.25 * (F_p[j - 1] - F_p[j + 1]) ** 2 +
                             (13 / 12) * (F_p[j - 1] - 2 * F_p[j] + F_p[j + 1]) ** 2)
                beta_p[2] = (0.25 * (3 * F_p[j] - 4 * F_p[j + 1] + F_p[j + 2]) ** 2 +
                             (13 / 12) * (F_p[j] - 2 * F_p[j + 1] + F_p[j + 2]) ** 2)

                # 计算权重
                alpha_p = C / (em + beta_p) ** p
                alpha_p_sum = np.sum(alpha_p)
                omega_p = alpha_p / alpha_p_sum

                # 计算三个模板的通量
                Fh_p_c = np.zeros(3)
                Fh_p_c[0] = (1 / 3) * F_p[j - 2] - (7 / 6) * F_p[j - 1] + (11 / 6) * F_p[j]
                Fh_p_c[1] = -(1 / 6) * F_p[j - 1] + (5 / 6) * F_p[j] + (1 / 3) * F_p[j + 1]
                Fh_p_c[2] = (1 / 3) * F_p[j] + (5 / 6) * F_p[j + 1] - (1 / 6) * F_p[j + 2]

                # 加权组合得到F+(j+1/2)
                Fh_p[j] = np.dot(omega_p, Fh_p_c)

                # === 负通量分量 (F-) ===
                # 计算光滑指示器 (方向反向)
                beta_n = np.zeros(3)
                beta_n[0] = (0.25 * (F_n[j + 2] - 4 * F_n[j + 1] + 3 * F_n[j]) ** 2 +
                             (13 / 12) * (F_n[j + 2] - 2 * F_n[j + 1] + F_n[j]) ** 2)
                beta_n[1] = (0.25 * (F_n[j + 1] - F_n[j - 1]) ** 2 +
                             (13 / 12) * (F_n[j + 1] - 2 * F_n[j] + F_n[j - 1]) ** 2)
                beta_n[2] = (0.25 * (3 * F_n[j] - 4 * F_n[j - 1] + F_n[j - 2]) ** 2 +
                             (13 / 12) * (F_n[j] - 2 * F_n[j - 1] + F_n[j - 2]) ** 2)

                # 计算权重
                alpha_n = C / (em + beta_n) ** p
                alpha_n_sum = np.sum(alpha_n)
                omega_n = alpha_n / alpha_n_sum

                # 计算三个模板的通量
                Fh_n_c = np.zeros(3)
                Fh_n_c[0] = (1 / 3) * F_n[j + 2] - (7 / 6) * F_n[j + 1] + (11 / 6) * F_n[j]
                Fh_n_c[1] = -(1 / 6) * F_n[j + 1] + (5 / 6) * F_n[j] + (1 / 3) * F_n[j - 1]
                Fh_n_c[2] = (1 / 3) * F_n[j] + (5 / 6) * F_n[j - 1] - (1 / 6) * F_n[j - 2]

                # 加权组合得到F-(j+1/2)
                Fh_n[j] = np.dot(omega_n, Fh_n_c)

            # 更新索引范围
            xs_new = xs + 1
            xt_new = xt - 1

            # 计算通量导数
            for j in range(xs_new, xt_new):
                Fx_p[j] = (Fh_p[j] - Fh_p[j - 1]) / dx
                Fx_n[j] = (Fh_n[j + 1] - Fh_n[j]) / dx
                Fx[j] = Fx_p[j] + Fx_n[j]

    return xs_new, xt_new, Fh_p, Fh_n, Fx, Fx_p, Fx_n

def Flux_Diff_Split_Common(U, N, dx, Gamma, Cp, Cv, R, flag_fds_met, flag_spa_typ, flag_upw_typ, flag_scs_typ):
    """
    通量差分裂方法(FDS)通用函数

    参数:
    U -- 守恒变量矩阵 (N×3)
    N -- 网格点数
    dx -- 网格尺寸
    Gamma -- 比热比
    Cp -- 定压比热容
    Cv -- 定容比热容
    R -- 气体常数
    flag_fds_met -- FDS方法标志
    flag_spa_typ -- 空间离散类型标志
    flag_upw_typ -- 迎风格式标志
    flag_scs_typ -- 激波捕捉格式标志

    返回:
    xs_new, xt_new, Fx -- 开始/结束索引和通量导数
    """
    if flag_fds_met == 1:
        # FDS - Roe格式
        # 初始化数组
        Fh_l = np.zeros((N, 3))
        Fh_r = np.zeros((N, 3))
        U_ave = np.zeros((N, 3))  # Roe平均U向量
        F_ave = np.zeros((N, 3))  # Roe平均F(U)向量
        Fh = np.zeros((N, 3))
        Fx = np.zeros((N, 3))
        A_ave = np.zeros((3, 3))  # Roe平均雅可比矩阵A(U)

        em = 1e-5  # 熵修正参数

        # 步骤1: 使用差分格式计算左右状态(Uh_l, Uh_r)
        # 调用差分函数计算左右守恒变量
        xs, xt, Uh_l, Uh_r, _, _, _ = Diff_Cons_Common(N, dx, U, U, flag_spa_typ, flag_upw_typ, flag_scs_typ)

        # 特殊情况: 处理迎风格式 & WENO格式 (F_l(j + 1/2), F_n(j - 1/2))
        if flag_spa_typ == 1 or (flag_spa_typ == 2 and flag_scs_typ == 3):
            xt = xt - 1

            # 将Uh_r从(j - 1/2)移动到(j + 1/2)
            for j in range(xs, xt + 1):
                Uh_r[j] = Uh_r[j + 1]  # Python索引从0开始

        # 循环计算每个网格点
        for j in range(xs, xt + 1):  # Python范围包含结尾

            # 提取左状态
            rho_l = Uh_l[j, 0]
            u_l = Uh_l[j, 1] / rho_l if rho_l != 0 else 0
            E_l = Uh_l[j, 2]
            # 计算温度和压力
            T_l = (E_l / rho_l - 0.5 * u_l ** 2) / Cv
            p_l = rho_l * R * T_l
            # 计算焓
            H_l = 0.5 * u_l ** 2 + Cp * T_l

            # 计算左通量F(Ul)
            Fh_l[j, 0] = rho_l * u_l
            Fh_l[j, 1] = rho_l * u_l ** 2 + p_l
            Fh_l[j, 2] = u_l * (E_l + p_l)

            # 提取右状态
            rho_r = Uh_r[j, 0]
            u_r = Uh_r[j, 1] / rho_r if rho_r != 0 else 0
            E_r = Uh_r[j, 2]
            # 计算温度和压力
            T_r = (E_r / rho_r - 0.5 * u_r ** 2) / Cv
            p_r = rho_r * R * T_r
            # 计算焓
            H_r = 0.5 * u_r ** 2 + Cp * T_r

            # 计算右通量F(Ur)
            Fh_r[j, 0] = rho_r * u_r
            Fh_r[j, 1] = rho_r * u_r ** 2 + p_r
            Fh_r[j, 2] = u_r * (E_r + p_r)

            # 步骤2: 计算Roe平均Ū
            rho_ave = ((np.sqrt(rho_l) + np.sqrt(rho_r)) / 2) ** 2
            # 避免除零
            if rho_ave != 0:
                u_ave = (np.sqrt(rho_l) * u_l + np.sqrt(rho_r) * u_r) / (2 * np.sqrt(rho_ave))
                H_ave = (np.sqrt(rho_l) * H_l + np.sqrt(rho_r) * H_r) / (2 * np.sqrt(rho_ave))
            else:
                u_ave = 0
                H_ave = 0

            # 计算平均压力和声速
            p_ave = (Gamma - 1) / Gamma * (rho_ave * H_ave - 0.5 * rho_ave * u_ave ** 2)
            c_ave = np.sqrt((Gamma - 1) * (H_ave - 0.5 * u_ave ** 2))
            E_ave = rho_ave * H_ave - p_ave

            # 存储平均状态
            U_ave[j, 0] = rho_ave
            U_ave[j, 1] = rho_ave * u_ave
            U_ave[j, 2] = E_ave

            # 计算平均通量F(Ū)
            F_ave[j, 0] = rho_ave * u_ave
            F_ave[j, 1] = rho_ave * u_ave ** 2 + p_ave
            F_ave[j, 2] = u_ave * (E_ave + p_ave)

            # 步骤3: 计算雅可比矩阵A(Ū)
            A_ave = np.zeros((3, 3))
            A_ave[0, :] = [0, 1, 0]
            A_ave[1, :] = [(-1) * ((3 - Gamma) / 2) * u_ave ** 2, (3 - Gamma) * u_ave, Gamma - 1]
            A_ave[2, :] = [
                ((Gamma - 2) / 2) * u_ave ** 3 - (u_ave * c_ave ** 2) / (Gamma - 1),
                c_ave ** 2 / (Gamma - 1) + ((3 - Gamma) / 2) * u_ave ** 2,
                Gamma * u_ave
            ]

            # 步骤4: 计算特征值和特征向量
            # 计算特征值和特征向量
            eigvals, V = np.linalg.eig(A_ave)
            # 计算特征向量的逆矩阵
            S = np.linalg.inv(V)
            # 创建特征值矩阵
            G = np.diag(eigvals)

            # 计算绝对值矩阵并进行熵修正
            G_abs = np.zeros((3, 3))
            for i in range(3):
                abs_val = abs(eigvals[i])
                if abs_val > em:
                    G_abs[i, i] = abs_val
                else:
                    G_abs[i, i] = (eigvals[i] ** 2 + em ** 2) / (2 * em)

            # 计算绝对雅可比矩阵 |A| = V·|Λ|·V⁻¹
            A_ave_abs = V @ G_abs @ S

            # 步骤5: 计算半网格点通量F(j+1/2)
            # Roe通量公式: F_{j+1/2} = 1/2[F_l + F_r] - 1/2 |A|(U_r - U_l)
            Fh[j] = 0.5 * (Fh_r[j] + Fh_l[j]) - 0.5 * A_ave_abs @ (Uh_r[j] - Uh_l[j])

        # 更新索引范围
        xs_new = xs + 1
        xt_new = xt

        # 步骤6: 计算通量导数Fx_j
        for j in range(xs_new, xt_new + 1):  # Python范围包含结尾
            # Fx = [F_{j+1/2} - F_{j-1/2}] / dx
            Fx[j] = (Fh[j] - Fh[j - 1]) / dx

    return xs_new, xt_new, Fx
def Flux_Vect_Split_Common(U, N, Gamma, Cp, Cv, R, flag_fvs_met):
    """
    通量向量分裂方法(FVS)通用函数

    参数:
    U -- 守恒变量矩阵 (N×3)
    N -- 网格点数
    Gamma -- 比热比
    Cp -- 定压比热容
    Cv -- 定容比热容
    R -- 气体常数
    flag_fvs_met -- FVS方法标志

    返回:
    F_p, F_n -- 正/负通量分量
    """
    if flag_fvs_met == 1:
        # 1 - FVS - Steger-Warming (S-W)方法

        # 初始化变量
        F_p = np.zeros((N, 3))  # 正通量分量
        F_n = np.zeros((N, 3))  # 负通量分量
        em = 1e-3  # 小量防止除零

        for i in range(N):
            # 步骤1: 计算密度、速度、温度、压力、声速
            rho = U[i, 0]
            u = U[i, 1] / rho
            E = U[i, 2]
            # 计算温度: T = (e_total - 0.5*u²) / Cv
            T = (E / rho - 0.5 * u ** 2) / Cv
            p = rho * R * T
            c = np.sqrt(Gamma * p / rho)

            # 步骤2: 计算特征值
            lambda_vals = np.array([u, u - c, u + c])

            # 步骤3: 分裂特征值 (S-W方法)
            sqrt_term = np.sqrt(lambda_vals ** 2 + em ** 2)
            lambda_p = (lambda_vals + sqrt_term) / 2
            lambda_n = (lambda_vals - sqrt_term) / 2

            # 步骤4: 计算F+和F-
            # 计算w项
            w_p = ((3 - Gamma) * (lambda_p[1] + lambda_p[2]) * c ** 2) / (2 * (Gamma - 1))
            w_n = ((3 - Gamma) * (lambda_n[1] + lambda_n[2]) * c ** 2) / (2 * (Gamma - 1))

            # 计算正通量分量
            F_p[i, 0] = (rho / (2 * Gamma)) * (2 * (Gamma - 1) * lambda_p[0] + lambda_p[1] + lambda_p[2])
            F_p[i, 1] = (rho / (2 * Gamma)) * (
                        2 * (Gamma - 1) * lambda_p[0] * u + lambda_p[1] * (u - c) + lambda_p[2] * (u + c))
            F_p[i, 2] = (rho / (2 * Gamma)) * ((Gamma - 1) * lambda_p[0] * u ** 2 + (lambda_p[1] * (u - c) ** 2) / 2 + (
                        lambda_p[2] * (u + c) ** 2) / 2 + w_p)

            # 计算负通量分量
            F_n[i, 0] = (rho / (2 * Gamma)) * (2 * (Gamma - 1) * lambda_n[0] + lambda_n[1] + lambda_n[2])
            F_n[i, 1] = (rho / (2 * Gamma)) * (
                        2 * (Gamma - 1) * lambda_n[0] * u + lambda_n[1] * (u - c) + lambda_n[2] * (u + c))
            F_n[i, 2] = (rho / (2 * Gamma)) * ((Gamma - 1) * lambda_n[0] * u ** 2 + (lambda_n[1] * (u - c) ** 2) / 2 + (
                        lambda_n[2] * (u + c) ** 2) / 2 + w_n)

    elif flag_fvs_met == 2:
        # 2 - FVS - Lax-Friedrich (L-F)方法

        # 初始化变量
        F_p = np.zeros((N, 3))  # 正通量分量
        F_n = np.zeros((N, 3))  # 负通量分量

        # 先循环计算全局最大特征值
        lambda_s_global = 0
        for i in range(N):
            rho = U[i, 0]
            u = U[i, 1] / rho
            E = U[i, 2]
            T = (E / rho - 0.5 * u ** 2) / Cv
            p = rho * R * T
            c = np.sqrt(Gamma * p / rho)

            # 计算局部特征值并更新全局最大值
            lambda_s_local = abs(u) + c
            if lambda_s_local > lambda_s_global:
                lambda_s_global = lambda_s_local

        # 再次循环计算通量分量
        for i in range(N):
            # 计算物理量
            rho = U[i, 0]
            u = U[i, 1] / rho
            E = U[i, 2]
            T = (E / rho - 0.5 * u ** 2) / Cv
            p = rho * R * T
            c = np.sqrt(Gamma * p / rho)

            # 计算特征值
            lambda_vals = np.array([u, u - c, u + c])

            # 分裂特征值 (L-F方法使用全局最大特征值)
            lambda_p = (lambda_vals + lambda_s_global) / 2
            lambda_n = (lambda_vals - lambda_s_global) / 2

            # 计算w项
            w_p = ((3 - Gamma) * (lambda_p[1] + lambda_p[2]) * c ** 2) / (2 * (Gamma - 1))
            w_n = ((3 - Gamma) * (lambda_n[1] + lambda_n[2]) * c ** 2) / (2 * (Gamma - 1))

            # 计算正通量分量
            F_p[i, 0] = (rho / (2 * Gamma)) * (2 * (Gamma - 1) * lambda_p[0] + lambda_p[1] + lambda_p[2])
            F_p[i, 1] = (rho / (2 * Gamma)) * (
                        2 * (Gamma - 1) * lambda_p[0] * u + lambda_p[1] * (u - c) + lambda_p[2] * (u + c))
            F_p[i, 2] = (rho / (2 * Gamma)) * ((Gamma - 1) * lambda_p[0] * u ** 2 + (lambda_p[1] * (u - c) ** 2) / 2 + (
                        lambda_p[2] * (u + c) ** 2) / 2 + w_p)

            # 计算负通量分量
            F_n[i, 0] = (rho / (2 * Gamma)) * (2 * (Gamma - 1) * lambda_n[0] + lambda_n[1] + lambda_n[2])
            F_n[i, 1] = (rho / (2 * Gamma)) * (
                        2 * (Gamma - 1) * lambda_n[0] * u + lambda_n[1] * (u - c) + lambda_n[2] * (u + c))
            F_n[i, 2] = (rho / (2 * Gamma)) * ((Gamma - 1) * lambda_n[0] * u ** 2 + (lambda_n[1] * (u - c) ** 2) / 2 + (
                        lambda_n[2] * (u + c) ** 2) / 2 + w_n)

    elif flag_fvs_met == 3:
        # 3 - Van Leer方法

        # 初始化变量
        F_p = np.zeros((N, 3))  # 正通量分量
        F_n = np.zeros((N, 3))  # 负通量分量

        for i in range(N):
            # 计算物理量
            rho = U[i, 0]
            u = U[i, 1] / rho
            E = U[i, 2]
            T = (E / rho - 0.5 * u ** 2) / Cv
            p = rho * R * T
            c = np.sqrt(Gamma * p / rho)

            # 计算马赫数
            Ma = u / c if c != 0 else 0

            # Van Leer通量分量计算公式
            if Ma >= 1:
                # 纯超音速正向流
                F_p[i, 0] = U[i, 1]  # rho * u
                F_p[i, 1] = (Gamma - 1) * U[i, 2] + ((3 - Gamma) / 2) * (U[i, 1] ** 2 / U[i, 0])
                F_p[i, 2] = Gamma * (U[i, 1] * U[i, 2]) / U[i, 0] + ((Gamma - 1) / 2) * (U[i, 1] ** 3 / U[i, 0] ** 2)

                F_n[i, :] = 0.0

            elif Ma <= -1:
                # 纯超音速负向流
                F_n[i, 0] = U[i, 1]  # rho * u
                F_n[i, 1] = (Gamma - 1) * U[i, 2] + ((3 - Gamma) / 2) * (U[i, 1] ** 2 / U[i, 0])
                F_n[i, 2] = Gamma * (U[i, 1] * U[i, 2]) / U[i, 0] + ((Gamma - 1) / 2) * (U[i, 1] ** 3 / U[i, 0] ** 2)

                F_p[i, :] = 0.0

            else:
                # 亚音速/跨音速区域
                # 计算质量通量
                F1_p = rho * c * ((Ma + 1) / 2) ** 2
                F1_n = -rho * c * ((Ma - 1) / 2) ** 2

                # 计算正通量分量
                F_p[i, 0] = F1_p
                F_p[i, 1] = (F1_p / Gamma) * ((Gamma - 1) * u + 2 * c)
                F_p[i, 2] = (F1_p / (2 * (Gamma ** 2 - 1))) * ((Gamma - 1) * u + 2 * c) ** 2

                # 计算负通量分量
                F_n[i, 0] = F1_n
                F_n[i, 1] = (F1_n / Gamma) * ((Gamma - 1) * u - 2 * c)
                F_n[i, 2] = (F1_n / (2 * (Gamma ** 2 - 1))) * ((Gamma - 1) * u - 2 * c) ** 2

    elif flag_fvs_met == 4:
        # 4 - Liou-Steffen AUSM方法

        # 初始化变量
        F_p = np.zeros((N, 3))  # 正通量分量
        F_n = np.zeros((N, 3))  # 负通量分量

        for i in range(N):
            # 计算物理量
            rho = U[i, 0]
            u = U[i, 1] / rho
            E = U[i, 2]
            T = (E / rho - 0.5 * u ** 2) / Cv
            p = rho * R * T
            c = np.sqrt(Gamma * p / rho)
            H = 0.5 * u ** 2 + Cp * T  # 焓

            # 计算马赫数
            Ma = u / c if c != 0 else 0

            # 计算分裂的马赫数和压力
            if Ma > 1:
                Ma_p = Ma
                Ma_n = 0
                p_p = p
                p_n = 0
            elif Ma < -1:
                Ma_p = 0
                Ma_n = Ma
                p_p = 0
                p_n = p
            else:
                # 亚音速情况
                Ma_p = (Ma + 1) ** 2 / 4
                Ma_n = -(Ma - 1) ** 2 / 4
                p_p = p * (1 + Ma) / 2
                p_n = p * (1 - Ma) / 2

            # 计算对流通量分量
            Fc_p = np.array([
                rho * c * Ma_p,
                rho * c * Ma_p * u,
                rho * c * Ma_p * H
            ])

            Fc_n = np.array([
                rho * c * Ma_n,
                rho * c * Ma_n * u,
                rho * c * Ma_n * H
            ])

            # 计算压力通量分量
            Fp_p = np.array([0, p_p, 0])
            Fp_n = np.array([0, p_n, 0])

            # 计算总通量分量
            F_p[i] = Fc_p + Fp_p
            F_n[i] = Fc_n + Fp_n

    return F_p, F_n
def plt_Head(filename, title, variables):
    """
    创建Tecplot文件头

    参数:
    filename -- Tecplot文件名
    title -- 标题字符串
    variables -- 变量名称列表
    """
    # 打开文件(追加模式)
    with open(filename, 'a') as f:
        # 添加标题
        if title:
            # 写入TITLE行: TITLE = "标题"
            s = f'TITLE = "{title}"'
            f.write(s + '\n')

        # 添加变量声明
        s = 'VARIABLES ='
        # 遍历所有变量
        for k, var in enumerate(variables):
            if k != 0:
                s += ','
            s += f' "{var}"'

        # 写入变量行
        f.write(s + '\n')
def Plot_Props(t, xp, rho, p, u, E, fig=None, axs=None):
    """
    绘制物理量随时间变化的图表

    参数:
    t -- 当前时间
    xp -- 位置坐标数组
    rho -- 密度数组
    p -- 压力数组
    u -- 速度数组
    E -- 比内能数组
    fig -- 图形对象(可选)
    axs -- 子图对象数组(可选)

    返回:
    fig -- 图形对象
    axs -- 子图对象数组
    """
    # 如果没有提供图形和子图，创建新的
    if fig is None or axs is None:
        fig, axs = plt.subplots(2, 2, figsize=(12, 8))
        plt.tight_layout(rect=[0, 0, 1, 0.95], pad=3.0)
        fig.suptitle(f'Sod Shock Tube Simulation (t = {t:.3f} s)', fontsize=16)

    # 设置四个子图的轴
    ax1, ax2, ax3, ax4 = axs.flatten()

    # 密度子图
    ax1.clear()
    ax1.plot(xp, rho, 'b-', linewidth=2)
    ax1.set_xlabel('Position (m)')
    ax1.set_ylabel('Density (kg/m³)')
    ax1.set_title('Density Profile')
    ax1.set_xlim(-0.5, 0.5)
    ax1.set_ylim(0.0, 1.0)
    ax1.grid(True)

    # 压力子图
    ax2.clear()
    ax2.plot(xp, p, 'g-', linewidth=2)
    ax2.set_xlabel('Position (m)')
    ax2.set_ylabel('Pressure (Pa)')
    ax2.set_title('Pressure Profile')
    ax2.set_xlim(-0.5, 0.5)
    ax2.set_ylim(0.0, 1.0)
    ax2.grid(True)

    # 速度子图
    ax3.clear()
    ax3.plot(xp, u, 'r-', linewidth=2)
    ax3.set_xlabel('Position (m)')
    ax3.set_ylabel('Velocity (m/s)')
    ax3.set_title('Velocity Profile')
    ax3.set_xlim(-0.5, 0.5)
    ax3.set_ylim(0.0, 1.0)
    ax3.grid(True)

    # 比内能子图
    ax4.clear()
    ax4.plot(xp, E, 'm-', linewidth=2)
    ax4.set_xlabel('Position (m)')
    ax4.set_ylabel('Specific Internal Energy (J/kg)')
    ax4.set_title('Specific Internal Energy Profile')
    ax4.set_xlim(-0.5, 0.5)
    ax4.set_ylim(1.5, 3.0)
    ax4.grid(True)

    # 更新主标题时间
    fig.suptitle(f'Sod Shock Tube Simulation (t = {t:.3f} s)', fontsize=16)

    # 自动调整布局
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    return fig, axs
def plt_Zone(filename, zone_title, IJK, Mat_Data):
    """
    创建Tecplot区域点数据格式

    参数:
    filename -- Tecplot文件名
    zone_title -- 区域标题
    IJK -- 网格尺寸数组或整数
    Mat_Data -- 数据矩阵
    """
    # 打开文件(追加模式)
    with open(filename, 'a') as f:
        N = len(Mat_Data)  # 数据点数

        # 判断维度
        if isinstance(IJK, int):
            # 一维点数据
            s = f'zone I={IJK}'
        elif len(IJK) == 1:
            # 一维点数据
            s = f'zone I={IJK[0]}'
        elif len(IJK) == 2:
            # 二维网格
            s = f'zone I={IJK[0]}, J={IJK[1]}'
        elif len(IJK) == 3:
            # 三维网格
            s = f'zone I={IJK[0]}, J={IJK[1]}, K={IJK[2]}'
        else:
            # 默认一维点数据
            s = f'zone I={len(Mat_Data)}'

        # 添加区域标题
        if zone_title:
            s += f', T="{zone_title}"'

        # 写入区域定义行
        f.write(s + '\n')

        # 指定数据打包格式为POINT
        f.write('DATAPACKING = POINT\n')

        # 写入数据点
        for k in range(N):
            # 将一行数据转换为空格分隔的字符串
            row_str = ' '.join([f'{val:.6f}' for val in Mat_Data[k]])
            f.write(row_str + '\n')
#%% 全局变量定义
def sod_func(P, gamma=1.4, rho_r=0.125, P_r=0.1):
    """Sod函数的Python实现"""
    mu_sq = (gamma - 1) / (gamma + 1)
    term = (1 - mu_sq) ** 2 / (rho_r * (P + mu_sq * P_r))
    return (P - P_r) * np.sqrt(term) - 2 * np.sqrt(gamma) / (gamma - 1) * (1 - P ** ((gamma - 1) / (2 * gamma)))
def analytic_sod(t=0.2):
    """
    计算Sod激波管问题的解析解

    参考: "http://www.phys.lsu.edu/~tohline/PHYS7412/sod.html"

    参数:
    t -- 时间(默认为0.2)

    返回:
    data -- 包含x, rho, P, u, e数组的字典
    """
    # 初始条件
    x0 = 0.0
    rho_l = 1.0
    P_l = 1.0
    u_l = 0.0

    rho_r = 0.125
    P_r = 0.1
    u_r = 0.0

    gamma = 1.4
    mu_sq = (gamma - 1) / (gamma + 1)

    # 计算声速
    c_l = np.sqrt(gamma * P_l / rho_l)
    c_r = np.sqrt(gamma * P_r / rho_r)

    # 求解 P_post (激波后的压力)
    P_post = fsolve(sod_func, np.pi, args=(gamma, rho_r, P_r))[0]

    # 计算中间状态量
    v_post = 2 * np.sqrt(gamma) / (gamma - 1) * (1 - P_post ** ((gamma - 1) / (2 * gamma)))
    rho_post = rho_r * ((P_post / P_r) + mu_sq) / (1 + mu_sq * (P_post / P_r))
    v_shock = v_post * (rho_post / rho_r) / ((rho_post / rho_r) - 1)
    rho_middle = rho_l * (P_post / P_l) ** (1 / gamma)

    # 计算关键位置
    x1 = x0 - c_l * t
    x3 = x0 + v_post * t
    x4 = x0 + v_shock * t
    c_2 = c_l - ((gamma - 1) / 2) * v_post
    x2 = x0 + (v_post - c_2) * t

    # 设置空间点
    n_points = 1000
    x_min = -0.5
    x_max = 0.5
    x = np.linspace(x_min, x_max, n_points)

    # 初始化数据数组
    data = {
        'x': x,
        'rho': np.zeros(n_points),
        'P': np.zeros(n_points),
        'u': np.zeros(n_points),
        'e': np.zeros(n_points)
    }

    # 计算每个点的物理量
    for i in range(n_points):
        x_val = x[i]

        if x_val < x1:
            # 区域 I: x < x1 (左波前)
            data['rho'][i] = rho_l
            data['P'][i] = P_l
            data['u'][i] = u_l

        elif x1 <= x_val <= x2:
            # 区域 II: x1 <= x <= x2 (膨胀波区)
            c = mu_sq * ((x0 - x_val) / t) + (1 - mu_sq) * c_l
            data['rho'][i] = rho_l * (c / c_l) ** (2 / (gamma - 1))
            data['P'][i] = P_l * (data['rho'][i] / rho_l) ** gamma
            data['u'][i] = (1 - mu_sq) * (-(x0 - x_val) / t + c_l)

        elif x2 <= x_val <= x3:
            # 区域 III: x2 <= x <= x3 (激波后的稳定区)
            data['rho'][i] = rho_middle
            data['P'][i] = P_post
            data['u'][i] = v_post

        elif x3 <= x_val <= x4:
            # 区域 IV: x3 <= x <= x4 (接触间断区)
            data['rho'][i] = rho_post
            data['P'][i] = P_post
            data['u'][i] = v_post

        else:  # x_val > x4
            # 区域 V: x > x4 (右波前)
            data['rho'][i] = rho_r
            data['P'][i] = P_r
            data['u'][i] = u_r

        # 计算比内能
        data['e'][i] = data['P'][i] / ((gamma - 1) * data['rho'][i])

    return data
# 管道特征长度 (x = [-L/2, L/2])
L = 1.0

# 流动参数
Gamma = 1.4  # 比热比
R = 286.9   # 气体常数 (J/kg·K)
Cv = R / (Gamma - 1)            # 定容比热容
Cp = (Gamma * R) / (Gamma - 1)  # 定压比热容

#%% 网格生成
N = 201   # x方向网格数
xp = np.linspace(-L / 2, L / 2, N)  # 网格点的x坐标
dx = L / (N - 1)  # 网格间距

xp_mid = N // 2  # x=0位置的网格索引(中点)

# 物理量数组初始化
u_arr = np.zeros(N)     # 速度数组
rho_arr = np.zeros(N)   # 密度数组
p_arr = np.zeros(N)     # 压力数组

# 辅助数组(可能冗余但保持原结构)
rho = np.zeros(N)
u = np.zeros(N)
p = np.zeros(N)
E = np.zeros(N)
T = np.zeros(N)
c = np.zeros(N)

# 时间步长(为稳定性可调整)
dt = 0.001

# 最大计算步数
max_step = 100

# 最大模拟时间
max_tot_time = max_step * dt

#%% 预处理：输入/输出文件设置
# # 设置全局输入/输出文件夹路径
# savefolder = f'Program_Sod_Shock_Tube_MaxTime_{max_tot_time:.3f}'  # 文件夹名称
# save_output_folder = os.path.join('.', savefolder, '')  # 完整路径
# os.makedirs(save_output_folder, exist_ok=True)  # 创建文件夹(如果不存在)

#%% 设置初始条件
# 当x < 0时，(左侧: ul, rhol, pl) = (0.0, 1.0, 1.0)
# 当x >= 0时，(右侧: ur, rhor, pr) = (0.0, 0.125, 0.1)
u_arr[0:xp_mid-1] = 0.0
rho_arr[0:xp_mid-1] = 1.0
p_arr[0:xp_mid-1] = 1.0

u_arr[xp_mid-1:] = 0.0    # Python使用0-based索引
rho_arr[xp_mid-1:] = 0.125
p_arr[xp_mid-1:] = 0.1

#%% 预处理：算法选择设置
# 使用枚举类型提高代码可读性
class FluxSplittingMethod(Enum):
    """通量分裂方法枚举"""
    FVS = 1  # 通量向量分裂
    FDS = 2  # 通量差分裂

class FVSMethods(Enum):
    """通量向量分裂方法枚举"""
    STEGER_WARMING = 1   # Steger-Warming方法
    LAX_FRIEDRICH = 2    # Lax-Friedrich方法
    VAN_LEER = 3         # Van Leer方法
    AUSM = 4             # AUSM方法

class FDSMethods(Enum):
    """通量差分裂方法枚举"""
    ROE = 1  # Roe格式

class SpatialDiscretizationType(Enum):
    """空间离散格式类型枚举"""
    UPWIND_COMPACT = 1   # 迎风/紧致格式
    SHOCK_CAPTURING = 2  # 激波捕捉格式

class UpwindSchemeType(Enum):
    """迎风格式类型枚举"""
    FIRST_ORDER = 1   # 一阶格式
    SECOND_ORDER = 2  # 二阶格式
    THIRD_ORDER = 3   # 三阶格式
    FIFTH_ORDER = 4   # 五阶格式

class ShockCapturingScheme(Enum):
    """激波捕捉格式枚举"""
    TVD_VAN_LEER = 1  # TVD格式(Van Leer限制器)
    NND = 2           # NND格式
    WENO = 3          # WENO格式

class TimeMarchingMethod(Enum):
    """时间推进方法枚举"""
    EULER = 1         # 欧拉法
    TRAPEZOIDAL = 2   # 梯形法
    RK2 = 3           # 二阶Runge-Kutta法
    TVD_RK3 = 4       # 三阶TVD Runge-Kutta法
    RK4 = 5           # 四阶Runge-Kutta法

# 指定通量分裂方法
# 1 - 通量向量分裂(FVS)
# 2 - 通量差分裂(FDS)
flag_flu_spl = FluxSplittingMethod.FVS.value  # 这里选择FVS方法

# FVS方法家族选择
# 1 - Steger-Warming (S-W)
# 2 - Lax-Friedrich (L-F)
# 3 - Van Leer
# 4 - Liou-Steffen分裂-AUSM方法
flag_fvs_met = FVSMethods.STEGER_WARMING.value  # 选择Steger-Warming方法

# 指定通量重构方法
# 1 - 直接重构(针对F(U))
# 2 - 特征重构
flag_flu_rec = 1  # 选择直接重构

# FDS方法家族选择(FDS-Roe格式)
flag_fds_met = FDSMethods.ROE.value

# 指定高分辨率通量空间离散格式
flag_spa_typ = SpatialDiscretizationType.UPWIND_COMPACT.value  # 迎风/紧致格式

# 迎风格式类型选择
flag_upw_typ = UpwindSchemeType.FIRST_ORDER.value  # 一阶迎风格式

# 激波捕捉格式类型选择
flag_scs_typ = ShockCapturingScheme.TVD_VAN_LEER.value  # TVD格式(Van Leer限制器)

# 指定时间推进方法
flag_tim_mar = TimeMarchingMethod.TVD_RK3.value  # 三阶TVD Runge-Kutta法

# # 区域标题设置(用于结果输出)
# zone_title_FVM = ["FVS", "FDS"]  # 通量分裂方法
# zone_title_FVS = ["S-W", "L-F", "Van Leer", "AUSM"]  # FVS方法
# zone_title_FDS = ["Roe"]  # FDS方法
# zone_title_SPA = ["UPW", "SCS"]  # 空间离散格式
# zone_title_UPW = ["1_od", "2_od", "3_od", "5_od"]  # 迎风格式
# zone_title_SCS = ["TVD (VL Limiter)", "NND", "WENO (5_od)"]  # 激波捕捉格式
# zone_title_MAR = ["Euler", "Trape", "R-K (2_od)", "R-K (3_od TVD)", "R-K (4_od)"]  # 时间推进方法
#
# # %% 根据算法选项生成组合标题字符串
# # 生成通量分裂方法组合标题
# if flag_flu_spl == 1:  # FVS方法
#     # 组合标题: "FVS_方法名"
#     zone_title_FVM_comb = f"{zone_title_FVM[flag_flu_spl - 1]}_{zone_title_FVS[flag_fvs_met - 1]}"
# elif flag_flu_spl == 2:  # FDS方法
#     # 组合标题: "FDS_方法名"
#     zone_title_FVM_comb = f"{zone_title_FVM[flag_flu_spl - 1]}_{zone_title_FDS[flag_fds_met - 1]}"
#
# # 生成空间离散方法组合标题
# if flag_spa_typ == 1:  # 迎风/紧致格式
#     # 组合标题: "UPW_格式名"
#     zone_title_SPA_comb = f"{zone_title_SPA[flag_spa_typ - 1]}_{zone_title_UPW[flag_upw_typ - 1]}"
# elif flag_spa_typ == 2:  # 激波捕捉格式
#     # 组合标题: "SCS_格式名"
#     zone_title_SPA_comb = f"{zone_title_SPA[flag_spa_typ - 1]}_{zone_title_SCS[flag_scs_typ - 1]}"
#
# # 生成时间推进方法标题
# zone_title_MAR_comb = zone_title_MAR[flag_tim_mar - 1]
#
# # 生成完整的组合标题
# zone_title_comb = f"{zone_title_FVM_comb}_{zone_title_SPA_comb}_{zone_title_MAR_comb}"
#
# # %% 导出动画设置
# flag_exp_mov = 0  # 是否导出视频文件标志
# flag_exp_avi = 0  # 是否导出AVI格式标志
# flag_exp_gif = 1  # 是否导出GIF动画标志

# %% 数组初始化
lambda_arr = np.zeros(3)  # 特征值矩阵
lambda_p_arr = np.zeros(3)  # 正特征值矩阵
lambda_n_arr = np.zeros(3)  # 负特征值矩阵

# 初始化解向量数组
U = np.zeros((N, 3))  # 守恒变量向量
F_p = np.zeros((N, 3))  # 正通量向量
F_n = np.zeros((N, 3))  # 负通量向量
Fx = np.zeros((N, 3))  # 通量散度项
Fx_p = np.zeros((N, 3))  # 正通量散度项
Fx_n = np.zeros((N, 3))  # 负通量散度项

# 半网格点通量数组
Fh_p = np.zeros((N - 1, 3))  # (j + 1/2)半网格点正通量向量
Fh_n = np.zeros((N - 1, 3))  # (j + 1/2)半网格点负通量向量

# %% 计算初始密度、速度、压力、温度、声速和守恒变量
for i in range(N):
    # 从数组获取当前网格点的值
    rho_val = rho_arr[i]
    u_val = u_arr[i]
    p_val = p_arr[i]

    # 计算温度 (理想气体状态方程)
    T_val = p_val / (rho_val * R)

    # 计算声速 (等熵关系)
    c_val = np.sqrt(Gamma * p_val / rho_val)

    # 计算总能量 (内能 + 动能)
    E_val = rho_val * ((Cv * T_val) + (0.5 * u_val * u_val))

    # 存储守恒变量
    # U = [密度, 动量, 总能量]
    U[i, 0] = rho_val  # 密度 (kg/m³)
    U[i, 1] = rho_val * u_val  # 动量 (kg/m²·s)
    U[i, 2] = E_val  # 总能量 (J/m³)

    # 存储热力学量
    T[i] = T_val
    c[i] = c_val

# # %% 动画设置
# # 创建帧存储结构（MATLAB的struct在Python中不直接对应）
# # F = struct('cdata',[],'colormap',[]);
# # Python中我们将使用字典模拟
# F = {'cdata': None, 'colormap': None}
#
# # 导出路径设置
# if flag_exp_avi == 1:
#     # 导入视频编码库
#     import matplotlib.animation as animation
#
#     # 准备视频文件路径
#     avi_path = os.path.join(save_output_folder, 'Sod_Shock_Tube.avi')
#
#     # 创建视频写入器对象
#     # 对应MATLAB的: avi_obj = VideoWriter([save_output_folder, 'Sod_Shock_Tube.avi'], 'Motion JPEG AVI');
#     # 使用MPEG-4编码，高质量
#     avi_writer = animation.FFMpegWriter(
#         fps=100,  # 帧率
#         codec='libx264',  # H.264编码
#         bitrate=-1,  # 自动选择码率
#         extra_args=['-crf', '18']  # 高质量设置
#     )
# %% 开始计算和时间步进
# 实际计算时间
t = 0.0
cnt_step = 0

# 主计算循环
while cnt_step < max_step:

    # 更新时间
    t += dt
    cnt_step += 1

    # 时间离散格式选择
    if flag_tim_mar == 1:
        # 1 - 欧拉时间步进法

        if flag_flu_spl == 1:
            # 1 - 通量向量分裂法 (FVS)
            F_p, F_n = Flux_Vect_Split_Common(U, N, Gamma, Cp, Cv, R, flag_fvs_met)
            xs_new, xt_new, Fh_p, Fh_n, Fx, Fx_p, Fx_n = Diff_Cons_Common(N, dx, F_p, F_n, flag_spa_typ, flag_upw_typ,
                                                                          flag_scs_typ)

        elif flag_flu_spl == 2:
            # 2 - 通量差分裂法 (FDS)
            xs_new, xt_new, Fx = Flux_Diff_Split_Common(U, N, dx, Gamma, Cp, Cv, R, flag_fds_met, flag_spa_typ,
                                                        flag_upw_typ, flag_scs_typ)

        # 欧拉更新: U^{n+1} = U^n + Δt * (-dF/dx)
        U = U + (dt * ((-1) * Fx))  # 右端项 Q(U) = (-1) * Fx

    elif flag_tim_mar == 2:
        # 2 - 梯形法 (二阶改进欧拉法)

        if flag_flu_spl == 1:
            # 1 - 通量向量分裂法 (FVS)
            F_p, F_n = Flux_Vect_Split_Common(U, N, Gamma, Cp, Cv, R, flag_fvs_met)
            _, _, _, _, Fx, _, _ = Diff_Cons_Common(N, dx, F_p, F_n, flag_spa_typ, flag_upw_typ, flag_scs_typ)

            # 第一步: 欧拉预估
            U_1 = U + (dt * ((-1) * Fx))

            # 使用预测值计算新通量
            F_p_1, F_n_1 = Flux_Vect_Split_Common(U_1, N, Gamma, Cp, Cv, R, flag_fvs_met)
            xs_new, xt_new, Fh_p, Fh_n, Fx_1, Fx_p, Fx_n = Diff_Cons_Common(N, dx, F_p_1, F_n_1, flag_spa_typ,
                                                                            flag_upw_typ, flag_scs_typ)

            # 梯形更新: U^{n+1} = 0.5*U^n + 0.5*U_1 + 0.5*Δt * Q(U_1)
            U = (0.5 * U) + (0.5 * U_1) + ((0.5 * dt) * ((-1) * Fx_1))

        elif flag_flu_spl == 2:
            # 2 - 通量差分裂法 (FDS)
            _, _, Fx = Flux_Diff_Split_Common(U, N, dx, Gamma, Cp, Cv, R, flag_fds_met, flag_spa_typ, flag_upw_typ,
                                              flag_scs_typ)

            # 第一步: 欧拉预估
            U_1 = U + (dt * ((-1) * Fx))

            # 使用预测值计算新通量
            xs_new, xt_new, Fx_1 = Flux_Diff_Split_Common(U_1, N, dx, Gamma, Cp, Cv, R, flag_fds_met, flag_spa_typ,
                                                          flag_upw_typ, flag_scs_typ)

            # 梯形更新
            U = (0.5 * U) + (0.5 * U_1) + ((0.5 * dt) * ((-1) * Fx_1))

    elif flag_tim_mar == 3:
        # 3 - 二阶龙格-库塔法 (Heun公式)

        if flag_flu_spl == 1:
            # 1 - 通量向量分裂法 (FVS)
            F_p, F_n = Flux_Vect_Split_Common(U, N, Gamma, Cp, Cv, R, flag_fvs_met)
            _, _, _, _, Fx, _, _ = Diff_Cons_Common(N, dx, F_p, F_n, flag_spa_typ, flag_upw_typ, flag_scs_typ)

            # 第一步: U1 = U^n + (Δt/3) * Q(U^n)
            U_1 = U + ((dt / 3) * ((-1) * Fx))

            # 使用U1计算Q(U1)
            F_p_1, F_n_1 = Flux_Vect_Split_Common(U_1, N, Gamma, Cp, Cv, R, flag_fvs_met)
            _, _, _, _, Fx_1, _, _ = Diff_Cons_Common(N, dx, F_p_1, F_n_1, flag_spa_typ, flag_upw_typ, flag_scs_typ)

            # 第二步: U2 = U^n + (2Δt/3) * Q(U1)
            U_2 = U + (((2 * dt) / 3) * ((-1) * Fx_1))

            # 使用U2计算Q(U2)
            F_p_2, F_n_2 = Flux_Vect_Split_Common(U_2, N, Gamma, Cp, Cv, R, flag_fvs_met)
            xs_new, xt_new, Fh_p, Fh_n, Fx_2, Fx_p, Fx_n = Diff_Cons_Common(N, dx, F_p_2, F_n_2, flag_spa_typ,
                                                                            flag_upw_typ, flag_scs_typ)

            # 最终更新: U^{n+1} = (1/4)U^n + (3/4)U1 + (3/4)Δt * Q(U2)
            U = ((1 / 4) * U) + ((3 / 4) * U_1) + (((3 / 4) * dt) * ((-1) * Fx_2))

        elif flag_flu_spl == 2:
            # 2 - 通量差分裂法 (FDS)
            _, _, Fx = Flux_Diff_Split_Common(U, N, dx, Gamma, Cp, Cv, R, flag_fds_met, flag_spa_typ, flag_upw_typ,
                                              flag_scs_typ)

            # 第一步
            U_1 = U + ((dt / 3) * ((-1) * Fx))
            _, _, Fx_1 = Flux_Diff_Split_Common(U_1, N, dx, Gamma, Cp, Cv, R, flag_fds_met, flag_spa_typ, flag_upw_typ,
                                                flag_scs_typ)

            # 第二步
            U_2 = U + (((2 * dt) / 3) * ((-1) * Fx_1))
            xs_new, xt_new, Fx_2 = Flux_Diff_Split_Common(U_2, N, dx, Gamma, Cp, Cv, R, flag_fds_met, flag_spa_typ,
                                                          flag_upw_typ, flag_scs_typ)

            # 最终更新
            U = ((1 / 4) * U) + ((3 / 4) * U_1) + (((3 / 4) * dt) * ((-1) * Fx_2))

    elif flag_tim_mar == 4:
        # 4 - 三阶TVD龙格-库塔法

        if flag_flu_spl == 1:
            # 1 - 通量向量分裂法 (FVS)
            F_p, F_n = Flux_Vect_Split_Common(U, N, Gamma, Cp, Cv, R, flag_fvs_met)
            _, _, _, _, Fx, _, _ = Diff_Cons_Common(N, dx, F_p, F_n, flag_spa_typ, flag_upw_typ, flag_scs_typ)

            # 第一步: U1 = U^n + Δt * Q(U^n)
            U_1 = U + (dt * ((-1) * Fx))

            # 使用U1计算Q(U1)
            F_p_1, F_n_1 = Flux_Vect_Split_Common(U_1, N, Gamma, Cp, Cv, R, flag_fvs_met)
            _, _, _, _, Fx_1, _, _ = Diff_Cons_Common(N, dx, F_p_1, F_n_1, flag_spa_typ, flag_upw_typ, flag_scs_typ)

            # 第二步: U2 = (3/4)U^n + (1/4)U1 + (1/4)Δt * Q(U1)
            U_2 = ((3 / 4) * U) + ((1 / 4) * U_1) + (((1 * dt) / 4) * ((-1) * Fx_1))

            # 使用U2计算Q(U2)
            F_p_2, F_n_2 = Flux_Vect_Split_Common(U_2, N, Gamma, Cp, Cv, R, flag_fvs_met)
            xs_new, xt_new, Fh_p, Fh_n, Fx_2, Fx_p, Fx_n = Diff_Cons_Common(N, dx, F_p_2, F_n_2, flag_spa_typ,
                                                                            flag_upw_typ, flag_scs_typ)

            # 最终更新: U^{n+1} = (1/3)U^n + (2/3)U2 + (2/3)Δt * Q(U2)
            U = ((1 / 3) * U) + ((2 / 3) * U_2) + (((2 / 3) * dt) * ((-1) * Fx_2))

        elif flag_flu_spl == 2:
            # 2 - 通量差分裂法 (FDS)
            _, _, Fx = Flux_Diff_Split_Common(U, N, dx, Gamma, Cp, Cv, R, flag_fds_met, flag_spa_typ, flag_upw_typ,
                                              flag_scs_typ)

            # 第一步
            U_1 = U + (dt * ((-1) * Fx))
            _, _, Fx_1 = Flux_Diff_Split_Common(U_1, N, dx, Gamma, Cp, Cv, R, flag_fds_met, flag_spa_typ, flag_upw_typ,
                                                flag_scs_typ)

            # 第二步
            U_2 = ((3 / 4) * U) + ((1 / 4) * U_1) + (((1 * dt) / 4) * ((-1) * Fx_1))
            xs_new, xt_new, Fx_2 = Flux_Diff_Split_Common(U_2, N, dx, Gamma, Cp, Cv, R, flag_fds_met, flag_spa_typ,
                                                          flag_upw_typ, flag_scs_typ)

            # 最终更新
            U = ((1 / 3) * U) + ((2 / 3) * U_2) + (((2 / 3) * dt) * ((-1) * Fx_2))

    elif flag_tim_mar == 5:
        # 5 - 四阶龙格-库塔法

        if flag_flu_spl == 1:
            # 1 - 通量向量分裂法 (FVS)
            F_p, F_n = Flux_Vect_Split_Common(U, N, Gamma, Cp, Cv, R, flag_fvs_met)
            _, _, _, _, Fx, _, _ = Diff_Cons_Common(N, dx, F_p, F_n, flag_spa_typ, flag_upw_typ, flag_scs_typ)

            # 第一步: U1 = U^n + (1/2)Δt * Q(U^n)
            U_1 = U + ((1 / 2) * (dt * ((-1) * Fx)))

            # 使用U1计算Q(U1)
            F_p_1, F_n_1 = Flux_Vect_Split_Common(U_1, N, Gamma, Cp, Cv, R, flag_fvs_met)
            _, _, _, _, Fx_1, _, _ = Diff_Cons_Common(N, dx, F_p_1, F_n_1, flag_spa_typ, flag_upw_typ, flag_scs_typ)

            # 第二步: U2 = U^n + (1/2)Δt * Q(U1)
            U_2 = U + ((1 / 2) * (dt * ((-1) * Fx_1)))

            # 使用U2计算Q(U2)
            F_p_2, F_n_2 = Flux_Vect_Split_Common(U_2, N, Gamma, Cp, Cv, R, flag_fvs_met)
            _, _, _, _, Fx_2, _, _ = Diff_Cons_Common(N, dx, F_p_2, F_n_2, flag_spa_typ, flag_upw_typ, flag_scs_typ)

            # 第三步: U3 = U^n + Δt * Q(U2)
            U_3 = U + (dt * ((-1) * Fx_2))

            # 使用U3计算Q(U3)
            F_p_3, F_n_3 = Flux_Vect_Split_Common(U_3, N, Gamma, Cp, Cv, R, flag_fvs_met)
            xs_new, xt_new, Fh_p, Fh_n, Fx_3, Fx_p, Fx_n = Diff_Cons_Common(N, dx, F_p_3, F_n_3, flag_spa_typ,
                                                                            flag_upw_typ, flag_scs_typ)

            # 最终更新: U^{n+1} = (1/3)[-U^n + U1 + 2U2 + U3] + (1/6)Δt * Q(U3)
            U = ((1 / 3) * (((-1) * U) + U_1 + (2 * U_2) + U_3)) + ((1 / 6) * dt * ((-1) * Fx_3))

        elif flag_flu_spl == 2:
            # 2 - 通量差分裂法 (FDS)
            _, _, Fx = Flux_Diff_Split_Common(U, N, dx, Gamma, Cp, Cv, R, flag_fds_met, flag_spa_typ, flag_upw_typ,
                                              flag_scs_typ)

            # 第一步
            U_1 = U + ((1 / 2) * (dt * ((-1) * Fx)))
            _, _, Fx_1 = Flux_Diff_Split_Common(U_1, N, dx, Gamma, Cp, Cv, R, flag_fds_met, flag_spa_typ, flag_upw_typ,
                                                flag_scs_typ)

            # 第二步
            U_2 = U + ((1 / 2) * (dt * ((-1) * Fx_1)))
            _, _, Fx_2 = Flux_Diff_Split_Common(U_2, N, dx, Gamma, Cp, Cv, R, flag_fds_met, flag_spa_typ, flag_upw_typ,
                                                flag_scs_typ)

            # 第三步
            U_3 = U + (dt * ((-1) * Fx_2))
            xs_new, xt_new, Fx_3 = Flux_Diff_Split_Common(U_3, N, dx, Gamma, Cp, Cv, R, flag_fds_met, flag_spa_typ,
                                                          flag_upw_typ, flag_scs_typ)

            # 最终更新
            U = ((1 / 3) * (((-1) * U) + U_1 + (2 * U_2) + U_3)) + ((1 / 6) * dt * ((-1) * Fx_3))

# if flag_exp_mov == 1:
#     # 保存时间t_final时刻的最终解U
#     U_tem = U.copy()  # 使用副本避免修改原数据
#
#     # 计算时间t_final时刻的最终物理量
#     # 根据守恒变量U计算rho, u, E, T, p, c, H
#     rho_tem = U_tem[:, 0]  # 密度
#     u_tem = U_tem[:, 1] / rho_tem  # 速度
#     E_tem = U_tem[:, 2]  # 单位质量总内能
#     # 计算温度: T = (e_total - 0.5*u²) / Cv
#     T_tem = (E_tem / rho_tem - 0.5 * u_tem ** 2) / Cv
#     # 计算压力: p = ρRT
#     p_tem = rho_tem * R * T_tem
#     # 计算声速: c = √(γp/ρ)
#     c_tem = np.sqrt(Gamma * p_tem / rho_tem)
#     # 计算焓: H = 0.5*u² + Cp*T
#     H_tem = 0.5 * u_tem ** 2 + Cp * T_tem
#     # 计算内能: e = E/ρ - 0.5*u²
#     e_tem = E_tem / rho_tem - 0.5 * u_tem ** 2
#
#     # data_tem = analytic_sod(t)  # 调用analytic_sod.m函数获取解析解(保留原函数名)
#
#     # 监测物理量
#     if cnt_step == 1:
#         # 创建图形对象
#         fig, ax = plt.subplots(3, 1, figsize=(10, 8))
#         fig.canvas.manager.set_window_title('Sod激波管模拟')
#         plt.tight_layout(pad=3.0)
#
#         # 调用绘图函数Plot_Props (保留原函数名)
#         Plot_Props(t, xp, rho_tem, p_tem, u_tem, e_tem, fig, ax)
#
#         # 捕获帧
#         if flag_exp_mov == 1:
#             fig.canvas.draw()
#             frame_data = np.array(fig.canvas.renderer.buffer_rgba())
#             F = {'cdata': frame_data, 'colormap': None}
#
#         # 导出视频
#         if flag_exp_avi == 1:
#             # 将帧添加到预创建的.avi文件句柄
#             avi_writer.grab_frame()
#
#         # 导出GIF图片
#         if flag_exp_gif == 1:
#             # 转换为PIL图像
#             im = Image.fromarray(frame_data)
#             # 转换为索引模式
#             if im.mode != 'P':
#                 im = im.convert('P', palette=Image.ADAPTIVE, colors=256)
#             # 保存GIF帧(使用原始MATLAB参数)
#             im.save(
#                 os.path.join(save_output_folder, 'Sod_Shock_Tube.gif'),
#                 format='GIF',
#                 save_all=False,
#                 loop=0,  # 无限循环
#                 append=True if cnt_step > 1 else False,
#                 duration=100  # 0.1秒帧率
#             )
#
#     else:
#         # 更新图形
#         Plot_Props(t, xp, rho_tem, p_tem, u_tem, e_tem, fig, ax)
#
#         # 强制刷新绘图
#         plt.draw()
#         plt.pause(0.001)
#
#         # 捕获帧
#         if flag_exp_mov == 1:
#             fig.canvas.draw()
#             frame_data = np.array(fig.canvas.renderer.buffer_rgba())
#             F = {'cdata': frame_data, 'colormap': None}
#
#         # 导出视频
#         if flag_exp_avi == 1:
#             # 将帧添加到预创建的.avi文件句柄
#             avi_writer.grab_frame()
#
#         # 导出GIF图片
#         if flag_exp_gif == 1:
#             # 转换为PIL图像
#             im = Image.fromarray(frame_data)
#             # 转换为索引模式
#             if im.mode != 'P':
#                 im = im.convert('P', palette=Image.ADAPTIVE, colors=256)
#             # 保存GIF帧(追加模式)
#             im.save(
#                 os.path.join(save_output_folder, 'Sod_Shock_Tube.gif'),
#                 format='GIF',
#                 save_all=False,
#                 append=True,  # 追加模式
#                 duration=100  # 0.1秒帧率
#             )
#%% 后处理

# # 关闭.avi文件句柄
# if flag_exp_avi == 1:
#
#     avi_writer.finish()  # 关闭并完成视频写入

# 保存时间t_end时刻的最终解U
U_end = U.copy()  # 使用副本
t_end = t

# 计算时间t_end时刻的最终物理量
# 根据守恒变量U计算rho, u, E, T, p, c, H
rho_end = U_end[:, 0]           # 密度
u_end = U_end[:, 1] / rho_end   # 速度
E_end = U_end[:, 2]             # 单位质量总内能
# 计算温度: T = (e_total - 0.5*u²) / Cv
T_end = (E_end / rho_end - 0.5 * u_end**2) / Cv
# 计算压力: p = ρRT
p_end = rho_end * R * T_end
# 计算声速: c = √(γp/ρ)
c_end = np.sqrt(Gamma * p_end / rho_end)
# 计算焓: H = 0.5*u² + Cp*T
H_end = 0.5 * u_end**2 + Cp * T_end
# 计算内能: e = E/ρ - 0.5*u²
e_end = E_end / rho_end - 0.5 * u_end**2

# 调用analytic_sod.m函数，获取解析解

data_end = analytic_sod(t_end)

#%% 可视化

# 创建最终结果图（添加背景色）
h_end = plt.figure(figsize=(10, 8), facecolor='white')  # 直接设置白色背景

# 使用变量存储子图对象便于管理
ax1 = plt.subplot(2, 2, 1)
ax1.set_title(f"密度分布 (时间 t = {t_end:.3f} s)")
ax1.plot(data_end.x, data_end.rho, '-b', linewidth=1.5, label='精确解')
ax1.plot(xp, rho_end, 'bo', markersize=3, label='数值解')
ax1.legend()
ax1.set_xlabel('位置 (m)')
ax1.set_ylabel('密度 (kg/m³)')
ax1.set_xlim(-0.5, 0.5)
ax1.set_ylim(0.0, 1.0)
ax1.grid(True)

ax2 = plt.subplot(2, 2, 2)
ax2.set_title(f"压力分布 (时间 t = {t_end:.3f} s)")
ax2.plot(data_end.x, data_end.P, '-g', linewidth=1.5, label='精确解')
ax2.plot(xp, p_end, 'go', markersize=3, label='数值解')
ax2.legend()
ax2.set_xlabel('位置 (m)')
ax2.set_ylabel('压力 (Pa)')
ax2.set_xlim(-0.5, 0.5)
ax2.set_ylim(0.0, 1.0)
ax2.grid(True)

ax3 = plt.subplot(2, 2, 3)
ax3.set_title(f"速度分布 (时间 t = {t_end:.3f} s)")
ax3.plot(data_end.x, data_end.u, '-r', linewidth=1.5, label='精确解')
ax3.plot(xp, u_end, 'ro', markersize=3, label='数值解')
ax3.legend()
ax3.set_xlabel('位置 (m)')
ax3.set_ylabel('速度 (m/s)')
ax3.set_xlim(-0.5, 0.5)
ax3.set_ylim(0.0, 1.0)
ax3.grid(True)

ax4 = plt.subplot(2, 2, 4)
ax4.set_title(f"比内能分布 (时间 t = {t_end:.3f} s)")
ax4.plot(data_end.x, data_end.e, '-m', linewidth=1.5, label='精确解')
ax4.plot(xp, e_end, 'mo', markersize=3, label='数值解')
ax4.legend()
ax4.set_xlabel('位置 (m)')
ax4.set_ylabel('比内能 (J/kg)')
ax4.set_xlim(-0.5, 0.5)
ax4.set_ylim(1.5, 3.0)
ax4.grid(True)

# 调整布局（增加主标题）
plt.suptitle(f"Sod激波管问题数值解与解析解对比 (t = {t_end:.3f}s)", fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.96])  # 为suptitle留出空间
plt.show()

# # 保存图形
# fig_save_path = os.path.join(save_output_folder, f'Results_{zone_title_comb}_MaxTime_{max_tot_time:.3f}.png')
# plt.savefig(fig_save_path, dpi=300)

# #%% 导出结果到Tecplot格式
#
# # 准备数值解数据
# title_cal = zone_title_comb
# zone_title_cal = zone_title_comb
# filename_cal = os.path.join(save_output_folder, f'Results_Sod_Shock_Tube_{zone_title_cal}_MaxTime_{max_tot_time:.3f}.plt')
# variables_cal = ['X', 'Density', 'Pressure', 'Velocity', 'Specific Internal Energy']
# Mat_Data_cal = np.column_stack([xp, rho_end, p_end, u_end, e_end])
# IJK_cal = len(xp)
#
# # 创建Tecplot文件并写入数值解
# # 如果文件已存在则删除
# if os.path.exists(filename_cal):
#     os.remove(filename_cal)
#
# # 写入Tecplot头信息和区域数据
# plt_Head(filename_cal, title_cal, variables_cal)  # 保留原函数名
# plt_Zone(filename_cal, zone_title_cal, IJK_cal, Mat_Data_cal)  # 保留原函数名
#
# # 准备精确解数据
# title_ana = 'Exact'
# zone_title_ana = 'Exact'
# filename_ana = os.path.join(save_output_folder, f'Results_Sod_Shock_Tube_{zone_title_ana}_MaxTime_{max_tot_time:.3f}.plt')
# variables_ana = ['X', 'Density', 'Pressure', 'Velocity', 'Specific Internal Energy']
# Mat_Data_ana = np.column_stack([data_end.x, data_end.rho, data_end.P, data_end.u, data_end.e])
# IJK_ana = len(data_end.x)
#
# # 创建Tecplot文件并写入精确解
# # 如果文件已存在则删除
# if os.path.exists(filename_ana):
#     os.remove(filename_ana)
#
# # 写入Tecplot头信息和区域数据
# plt_Head(filename_ana, title_ana, variables_ana)  # 保留原函数名
# plt_Zone(filename_ana, zone_title_ana, IJK_ana, Mat_Data_ana)  # 保留原函数名

#%% 保存程序变量
# # 对应MATLAB的save命令
# save_path = os.path.join(save_output_folder, f'Results_Variables_{zone_title_comb}_MaxTime_{max_tot_time:.3f}.mat')
# save_data = {
#     't_end': t_end,
#     'U_end': U_end,
#     'xp': xp,
#     'rho_end': rho_end,
#     'p_end': p_end,
#     'u_end': u_end,
#     'e_end': e_end,
#     'data_end': data_end,
#     'zone_title_comb': zone_title_comb,
#     'max_tot_time': max_tot_time
# }

# # 使用scipy保存为MAT格式
# from scipy.io import savemat
# savemat(save_path, save_data)

# 显示完成信息
# print(f"计算结果已保存到: {save_path}")


