import numpy as np
import matplotlib.pyplot as plt
from enum import Enum
from scipy.optimize import fsolve

#Minmod限制器函数
def Cal_Minmod(a, b):
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
#通用通量差分计算函数(TVD,WENO,GVC)，返回索引和通量导数
def Diff_Cons_Common(N, dx, F_p, F_n, flag_spa_typ, flag_scs_typ):
    # 初始化输出数组
    xs_new = 0
    xt_new = 0
    Fx = np.zeros((N, 3))  # 总通量导数
    Fx_p = np.zeros((N, 3))  # 正通量导数
    Fx_n = np.zeros((N, 3))  # 负通量导数
    Fh_p = np.zeros((N, 3))  # 半网格点正通量 (j+1/2)
    Fh_n = np.zeros((N, 3))  # 半网格点负通量 (j+1/2)
    Fh = np.zeros((N, 3))  # 半网格点总通量 (j+1/2)

    if flag_spa_typ == 1:
        #  特殊激波捕捉格式

        if flag_scs_typ == 1:
            # 1 - TVD格式 (Van Leer限制器)
            xs = 1  # 起始索引
            xt = N - 2  # 结束索引
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
            # 2 - WENO5阶格式
            C = np.array([1 / 10, 6 / 10, 3 / 10])
            p = 2
            em = 1e-6

            xs = 2  # 起始索引
            xt = N - 3  # 结束索引

            for j in range(xs, xt + 1):
                # a > 0 正通量部分
                # 计算光滑指示器 beta
                beta_p1 = (1 / 4) * (F_p[j - 2] - 4 * F_p[j - 1] + 3 * F_p[j]) ** 2 + (13 / 12) * (
                            F_p[j - 2] - 2 * F_p[j - 1] + F_p[j]) ** 2
                beta_p2 = (1 / 4) * (F_p[j - 1] - F_p[j + 1]) ** 2 + (13 / 12) * (F_p[j - 1] - 2 * F_p[j] + F_p[j + 1]) ** 2
                beta_p3 = (1 / 4) * (3 * F_p[j] - 4 * F_p[j + 1] + F_p[j + 2]) ** 2 + (13 / 12) * (
                            F_p[j] - 2 * F_p[j + 1] + F_p[j + 2]) ** 2

                # 计算 alpha 和权重 omega
                alpha_p = C / (em + np.array([beta_p1, beta_p2, beta_p3])) ** p
                alpha_p_sum = np.sum(alpha_p, axis=0)
                omega_p = alpha_p / alpha_p_sum

                # 计算三个模板的通量值
                Fh_p_c1 = (1 / 3) * F_p[j - 2] - (7 / 6) * F_p[j - 1] + (11 / 6) * F_p[j]
                Fh_p_c2 = (-1 / 6) * F_p[j - 1] + (5 / 6) * F_p[j] + (1 / 3) * F_p[j + 1]
                Fh_p_c3 = (1 / 3) * F_p[j] + (5 / 6) * F_p[j + 1] - (1 / 6) * F_p[j + 2]
                Fh_p_c = np.array([Fh_p_c1, Fh_p_c2, Fh_p_c3])

                # 加权组合得到 F+_{j+1/2}
                Fh_p[j] = np.sum(omega_p * Fh_p_c, axis=0)

                # a < 0 负通量部分
                # 计算光滑指示器 beta
                beta_n1 = (1 / 4) * (F_n[j + 2] - 4 * F_n[j + 1] + 3 * F_n[j]) ** 2 + (13 / 12) * (
                            F_n[j + 2] - 2 * F_n[j + 1] + F_n[j]) ** 2
                beta_n2 = (1 / 4) * (F_n[j + 1] - F_n[j - 1]) ** 2 + (13 / 12) * (F_n[j + 1] - 2 * F_n[j] + F_n[j - 1]) ** 2
                beta_n3 = (1 / 4) * (3 * F_n[j] - 4 * F_n[j - 1] + F_n[j - 2]) ** 2 + (13 / 12) * (
                            F_n[j] - 2 * F_n[j - 1] + F_n[j - 2]) ** 2

                # 计算 alpha 和权重 omega
                alpha_n = C / (em + np.array([beta_n1, beta_n2, beta_n3])) ** p
                alpha_n_sum = np.sum(alpha_n, axis=0)
                omega_n = alpha_n / alpha_n_sum

                # 计算三个模板的通量值
                Fh_n_c1 = (1 / 3) * F_n[j + 2] - (7 / 6) * F_n[j + 1] + (11 / 6) * F_n[j]
                Fh_n_c2 = (-1 / 6) * F_n[j + 1] + (5 / 6) * F_n[j] + (1 / 3) * F_n[j - 1]
                Fh_n_c3 = (1 / 3) * F_n[j] + (5 / 6) * F_n[j - 1] - (1 / 6) * F_n[j - 2]
                Fh_n_c = np.array([Fh_n_c1, Fh_n_c2, Fh_n_c3])

                # 加权组合得到 F-_{j-1/2}
                Fh_n[j] = np.sum(omega_n * Fh_n_c, axis=0)

            # 计算通量导数
            xs_new = xs + 1  # 3
            xt_new = xt - 1  # N-4

            for j in range(xs_new, xt_new + 1):
                Fx_p[j] = (Fh_p[j] - Fh_p[j - 1]) / dx
                Fx_n[j] = (Fh_n[j + 1] - Fh_n[j]) / dx
                Fx[j] = Fx_p[j] + Fx_n[j]

        elif flag_scs_typ == 3:
            # 群速度控制格式
            # 参数设置
            xs = 1  # 起始索引
            xt = N - 2  # 结束索引
            em = 1e-5  # 小量防止除零

            # 群速度控制参数
            beta = 0.8  # 群速度控制因子
            gamma = 0.3  # 高阶修正系数

            for j in range(xs, xt):
                # 计算通量梯度比
                # 正通量部分
                r_p_num = F_p[j] - F_p[j - 1]
                r_p_den = F_p[j + 1] - F_p[j] + em
                r_p = r_p_num / r_p_den

                # 负通量部分
                r_n_num = F_n[j + 2] - F_n[j + 1]
                r_n_den = F_n[j + 1] - F_n[j] + em
                r_n = r_n_num / r_n_den

                # 群速度控制限制器函数
                # 正通量限制器
                Phi_p = (r_p + beta * np.abs(r_p)) / (1 + beta * r_p + gamma * r_p ** 2)
                # 负通量限制器
                Phi_n = (r_n + beta * np.abs(r_n)) / (1 + beta * r_n + gamma * r_n ** 2)

                # 计算半网格点通量（含群速度控制）
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
    return xs_new, xt_new, Fh_p, Fh_n, Fx, Fx_p, Fx_n

#使用 ROE 格式实现通量差分裂法 (FDS)
def Flux_Diff_Split_Common(U, N, dx, Gamma, Cp, Cv, R, flag_fds_met, flag_spa_typ, flag_scs_typ):
    # Roe格式通量差分分裂
    if flag_fds_met == 1:
        # 初始化数组
        Fh_l = np.zeros((N, 3))  # 左界面通量
        Fh_r = np.zeros((N, 3))  # 右界面通量
        U_ave = np.zeros((N, 3))  # Roe平均守恒变量
        F_ave = np.zeros((N, 3))  # Roe平均通量
        Fh = np.zeros((N, 3))  # 界面通量
        Fx = np.zeros((N, 3))  # 空间导数

        em = 1e-5  # 熵修正参数

        # 调用守恒变量差分函数
        xs, xt, Uh_l, Uh_r, _, _, _ = Diff_Cons_Common(
            N, dx, U, U, flag_spa_typ, 2
        )

        # WENO格式
        if flag_spa_typ == 1 and flag_scs_typ == 2:
            xt = xt - 1  # 调整计算域结束索引
            # 右状态值向右平移一位
            for j in range(xs, xt + 1):
                Uh_r[j] = Uh_r[j + 1]

        # 遍历所有网格界面
        for j in range(xs, xt + 1):
            # --- 计算左状态物理量 ---
            rho_l = Uh_l[j, 0]  # 左侧密度
            u_l = Uh_l[j, 1] / rho_l  # 左侧速度
            E_l = Uh_l[j, 2]  # 左侧总能
            # 左侧温度
            T_l = ((E_l / rho_l) - 0.5 * u_l ** 2) / Cv
            p_l = rho_l * R * T_l  # 左侧压力
            H_l = 0.5 * u_l ** 2 + Cp * T_l  # 左侧总焓

            # 左侧通量计算
            Fh_l[j, 0] = rho_l * u_l  # 质量通量
            Fh_l[j, 1] = rho_l * u_l ** 2 + p_l  # 动量通量
            Fh_l[j, 2] = u_l * (E_l + p_l)  # 能量通量

            # --- 计算右状态物理量 ---
            rho_r = Uh_r[j, 0]  # 右侧密度
            u_r = Uh_r[j, 1] / rho_r  # 右侧速度
            E_r = Uh_r[j, 2]  # 右侧总能
            # 右侧温度
            T_r = ((E_r / rho_r) - 0.5 * u_r ** 2) / Cv
            p_r = rho_r * R * T_r  # 右侧压力
            H_r = 0.5 * u_r ** 2 + Cp * T_r  # 右侧总焓

            # 右侧通量计算
            Fh_r[j, 0] = rho_r * u_r  # 质量通量
            Fh_r[j, 1] = rho_r * u_r ** 2 + p_r  # 动量通量
            Fh_r[j, 2] = u_r * (E_r + p_r)  # 能量通量

            # --- 计算Roe平均量 ---
            sqrt_rho_l = np.sqrt(rho_l)
            sqrt_rho_r = np.sqrt(rho_r)
            # Roe平均密度
            rho_ave = ((sqrt_rho_l + sqrt_rho_r) / 2) ** 2
            # Roe平均速度
            u_ave = (sqrt_rho_l * u_l + sqrt_rho_r * u_r) / (np.sqrt(rho_ave) * 2)
            # Roe平均总焓
            H_ave = (sqrt_rho_l * H_l + sqrt_rho_r * H_r) / (np.sqrt(rho_ave) * 2)
            # Roe平均压力
            p_ave = ((Gamma - 1) / Gamma) * (rho_ave * H_ave - 0.5 * rho_ave * u_ave ** 2)
            # Roe平均声速
            c_ave = np.sqrt((Gamma - 1) * (H_ave - 0.5 * u_ave ** 2))
            # Roe平均总能
            E_ave = rho_ave * H_ave - p_ave

            # 存储Roe平均守恒变量
            U_ave[j] = [rho_ave, rho_ave * u_ave, E_ave]
            # 存储Roe平均通量
            F_ave[j] = [rho_ave * u_ave, rho_ave * u_ave ** 2 + p_ave, u_ave * (E_ave + p_ave)]

            # --- 构造Jacobian矩阵 ---
            A_ave = np.array([
                [0, 1, 0],
                [(-(3 - Gamma) / 2) * u_ave ** 2, (3 - Gamma) * u_ave, Gamma - 1],
                [((Gamma - 2) / 2) * u_ave ** 3 - u_ave * c_ave ** 2 / (Gamma - 1),
                 c_ave ** 2 / (Gamma - 1) + (3 - Gamma) / 2 * u_ave ** 2,
                 Gamma * u_ave]
            ])

            # 特征分解
            D, V = np.linalg.eig(A_ave)
            S = np.linalg.inv(V)

            # 熵修正处理特征值
            D_abs = np.zeros(3)
            for i in range(3):
                if abs(D[i]) > em:
                    D_abs[i] = abs(D[i])
                else:
                    # Harten修正公式
                    D_abs[i] = (D[i] ** 2 + em ** 2) / (2 * em)

            # 构造绝对Jacobian矩阵
            A_ave_abs = V @ np.diag(D_abs) @ S

            # 计算界面通量
            diff = (Uh_r[j] - Uh_l[j]).reshape(3, 1)  # 状态差向量
            # Roe通量公式
            roe_flux = 0.5 * (A_ave_abs @ diff).flatten()
            Fh[j] = 0.5 * (Fh_l[j] + Fh_r[j]) - roe_flux

        #计算空间导数
        xs_new = xs + 1  # 新计算域起始索引
        xt_new = xt  # 新计算域结束索引
        for j in range(xs_new, xt_new + 1):
            # 中心差分格式
            Fx[j] = (Fh[j] - Fh[j - 1]) / dx

    return xs_new, xt_new, Fx
#通量向量分裂方法(FVS)通用函数(Steger-Warming,Lax-Friedrich,Van Leer)
def Flux_Vect_Split_Common(U, N, Gamma, Cp, Cv, R, flag_fvs_met):
    if flag_fvs_met == 1:
        #  FVS - Steger-Warming (S-W)方法

        # 初始化变量
        F_p = np.zeros((N, 3))  # 正通量分量
        F_n = np.zeros((N, 3))  # 负通量分量
        em = 1e-3  # 小量防止除零

        for i in range(N):
            # 计算密度、速度、温度、压力、声速
            rho = U[i, 0]
            u = U[i, 1] / rho
            E = U[i, 2]
            # 计算温度: T = (e_total - 0.5*u²) / Cv
            T = (E / rho - 0.5 * u ** 2) / Cv
            p = rho * R * T
            c = np.sqrt(Gamma * p / rho)

            # 计算特征值
            lambda_vals = np.array([u, u - c, u + c])

            # 分裂特征值 (S-W方法)
            sqrt_term = np.sqrt(lambda_vals ** 2 + em ** 2)
            lambda_p = (lambda_vals + sqrt_term) / 2
            lambda_n = (lambda_vals - sqrt_term) / 2

            # 计算F+和F-
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
        # FVS - Lax-Friedrich (L-F)方法

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
        # Van Leer方法

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

    return F_p, F_n
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

#计算sod激波管解析解，源自https://github.com/sbakkerm/Sod-Shock-Tube/tree/main
def solve(left_state=(1, 1, 0), right_state=(0.1, 0.125, 0.), geometry=(-0.5, 0.5, 0), t=0.2, **kwargs):
    """
    Solves the Sod shock tube problem (i.e. riemann problem) of discontinuity across an interface.

    :param left_state: tuple (pl, rhol, ul)
    :param right_state: tuple (pr, rhor, ur)
    :param geometry: tuple (xl, xr, xi): xl - left boundary, xr - right boundary, xi - initial discontinuity
    :param t: time for which the states have to be calculated
    :param gamma: ideal gas constant, default is air: 1.4
    :param npts: number of points for array of pressure, density and velocity
    :param dustFrac: dust to gas fraction, should be >=0 and <1
    :return: tuple of: dicts of positions, regions, and val_dict with arrays of x, p, rho, u
    """

    if 'npts' in kwargs:
        npts = kwargs['npts']
    else:
        npts = 500

    if 'gamma' in kwargs:
        gamma = kwargs['gamma']
    else:
        gamma = 1.4

    if 'dustFrac' in kwargs:
        dustFrac = kwargs['dustFrac']
        if dustFrac < 0 or dustFrac >= 1:
            print('Invalid dust fraction value: {}. Should be >=0 and <1. Set to default: 0'.format(dustFrac))
            dustFrac = 0
    else:
        dustFrac = 0

    calculator = Calculator(left_state=left_state, right_state=right_state, geometry=geometry, t=t,
                            gamma=gamma, npts=npts, dustFrac=dustFrac)

    return calculator.solve()
class Calculator:
    """
    Class that does the actual work computing the Sod shock tube problem
    """

    def __init__(self, left_state, right_state, geometry, t, **kwargs):
        """
        Constructor
        :param left_state: tuple (pl, rhol, ul)
        :param right_state: tuple (pr, rhor, ur)
        :param geometry: tuple (xl, xr, xi): xl - left boundary, xr - right boundary, xi - initial discontinuity
        :param t: time for which the states have to be calculated
        :param gamma: ideal gas constant, default is air: 1.4
        :param npts: number of points for array of pressure, density and velocity
        :param dustFrac: dust fraction
        """
        self.pl, self.rhol, self.ul = left_state
        self.pr, self.rhor, self.ur = right_state
        self.xl, self.xr, self.xi = geometry
        self.t = t

        if 'npts' in kwargs:
            self.npts = kwargs['npts']
        else:
            self.npts = 500

        if 'gamma' in kwargs:
            self.gamma = kwargs['gamma']
        else:
            self.gamma = 1.4

        if 'dustFrac' in kwargs:
            self.dustFrac = kwargs['dustFrac']
        else:
            self.dustFrac = 0

        # basic checking
        if self.xl >= self.xr:
            print('xl has to be less than xr!')
            exit()
        if self.xi >= self.xr or self.xi <= self.xl:
            print('xi has in between xl and xr!')
            exit()

        # calculate regions
        self.region1, self.region3, self.region4, self.region5, self.w = \
            self.calculate_regions()

    def solve(self):
        """
        Actually solves the sod shock tube problem
        :return: positions, regions, val_dict
        """
        regions = self.region_states()

        # calculate positions
        x_positions = self.calc_positions()

        pos_description = ('Head of Rarefaction', 'Foot of Rarefaction',
                           'Contact Discontinuity', 'Shock')
        positions = dict(zip(pos_description, x_positions))

        # create arrays
        x, p, rho, u = self.create_arrays(x_positions)

        val_names = ('x', 'p', 'rho', 'u')
        val_dict = dict(zip(val_names, (x, p, rho, u)))

        return positions, regions, val_dict

    def sound_speed(self, p, rho):
        """
        Calculate speed of sound considering dust fraction
        """
        return np.sqrt(self.gamma * (1 - self.dustFrac) * p / rho)

    def shock_tube_function(self, p4, p1, p5, rho1, rho5):
        """
        Shock tube equation
        """
        z = (p4 / p5 - 1.)
        c1 = self.sound_speed(p1, rho1)
        c5 = self.sound_speed(p5, rho5)

        gm1 = self.gamma - 1.
        gp1 = self.gamma + 1.
        g2 = 2. * self.gamma

        fact = gm1 / g2 * (c5 / c1) * z / np.sqrt(1. + gp1 / g2 * z)
        fact = (1. - fact) ** (g2 / gm1)

        return p1 * fact - p4

    def calculate_regions(self):
        """
        Compute regions
        :return: p, rho and u for regions 1,3,4,5 and shock speed
        """
        # if pl > pr...
        rho1 = self.rhol
        p1 = self.pl
        u1 = self.ul
        rho5 = self.rhor
        p5 = self.pr
        u5 = self.ur

        # unless...
        if self.pl < self.pr:
            rho1 = self.rhor
            p1 = self.pr
            u1 = self.ur
            rho5 = self.rhol
            p5 = self.pl
            u5 = self.ul

        # solve for post-shock pressure
        num_of_guesses = 100
        pressure_range = np.linspace(min(self.pr, self.pl), max(self.pr, self.pl), num_of_guesses)

        ier = 0
        for pguess in pressure_range:
            res = fsolve(self.shock_tube_function, pguess, args=(p1, p5, rho1, rho5), full_output=True)
            p4, infodict, ier, mesg = res
            if ier == 1:
                break
        if not ier == 1:
            raise Exception("Analytical Sod solution unsuccessful!")

        if type(p4) is np.ndarray:
            p4 = p4[0]

        # compute post-shock density and velocity
        z = (p4 / p5 - 1.)
        c5 = self.sound_speed(p5, rho5)

        gm1 = self.gamma - 1.
        gp1 = self.gamma + 1.
        gmfac1 = 0.5 * gm1 / self.gamma
        gmfac2 = 0.5 * gp1 / self.gamma

        fact = np.sqrt(1. + gmfac2 * z)

        u4 = c5 * z / (self.gamma * fact)
        rho4 = rho5 * (1. + gmfac2 * z) / (1. + gmfac1 * z)

        # shock speed
        w = c5 * fact

        # compute values at foot of rarefaction
        p3 = p4
        u3 = u4
        rho3 = rho1 * (p3 / p1) ** (1. / self.gamma)
        return (p1, rho1, u1), (p3, rho3, u3), (p4, rho4, u4), (p5, rho5, u5), w

    def region_states(self):
        """
        :return: dictionary of regions
        """
        if self.pl > self.pr:
            return {'Region 1': self.region1,
                    'Region 2': 'RAREFACTION',
                    'Region 3': self.region3,
                    'Region 4': self.region4,
                    'Region 5': self.region5}
        else:
            return {'Region 1': self.region5,
                    'Region 2': self.region4,
                    'Region 3': self.region3,
                    'Region 4': 'RAREFACTION',
                    'Region 5': self.region1}

    def calc_positions(self):
        """
        :return: tuple of positions (head of rarefaction, foot of rarefaction, contact discontinuity, shock)
        """
        p1, rho1 = self.region1[:2]  # don't need velocity
        p3, rho3, u3 = self.region3[:]
        c1 = self.sound_speed(p1, rho1)
        c3 = self.sound_speed(p3, rho3)
        if self.pl > self.pr:
            xsh = self.xi + self.w * self.t
            xcd = self.xi + u3 * self.t
            xft = self.xi + (u3 - c3) * self.t
            xhd = self.xi - c1 * self.t
        else:
            # pr > pl
            xsh = self.xi - self.w * self.t
            xcd = self.xi - u3 * self.t
            xft = self.xi - (u3 - c3) * self.t
            xhd = self.xi + c1 * self.t

        return xhd, xft, xcd, xsh

    def create_arrays(self, positions):
        """
        :return: arrays of x, p, rho and u values
        """
        xhd, xft, xcd, xsh = positions
        p1, rho1, u1 = self.region1
        p3, rho3, u3 = self.region3
        p4, rho4, u4 = self.region4
        p5, rho5, u5 = self.region5
        gm1 = self.gamma - 1.
        gp1 = self.gamma + 1.

        x_arr = np.linspace(self.xl, self.xr, self.npts)
        rho = np.zeros(self.npts, dtype=float)
        p = np.zeros(self.npts, dtype=float)
        u = np.zeros(self.npts, dtype=float)
        c1 = self.sound_speed(p1, rho1)

        if self.pl > self.pr:
            for i, x in enumerate(x_arr):
                if x < xhd:
                    rho[i] = rho1
                    p[i] = p1
                    u[i] = u1
                elif x < xft:
                    u[i] = 2. / gp1 * (c1 + (x - self.xi) / self.t)
                    fact = 1. - 0.5 * gm1 * u[i] / c1
                    rho[i] = rho1 * fact ** (2. / gm1)
                    p[i] = p1 * fact ** (2. * self.gamma / gm1)
                elif x < xcd:
                    rho[i] = rho3
                    p[i] = p3
                    u[i] = u3
                elif x < xsh:
                    rho[i] = rho4
                    p[i] = p4
                    u[i] = u4
                else:
                    rho[i] = rho5
                    p[i] = p5
                    u[i] = u5
        else:
            for i, x in enumerate(x_arr):
                if x < xsh:
                    rho[i] = rho5
                    p[i] = p5
                    u[i] = -u1
                elif x < xcd:
                    rho[i] = rho4
                    p[i] = p4
                    u[i] = -u4
                elif x < xft:
                    rho[i] = rho3
                    p[i] = p3
                    u[i] = -u3
                elif x < xhd:
                    u[i] = -2. / gp1 * (c1 + (self.xi - x) / self.t)
                    fact = 1. + 0.5 * gm1 * u[i] / c1
                    rho[i] = rho1 * fact ** (2. / gm1)
                    p[i] = p1 * fact ** (2. * self.gamma / gm1)
                else:
                    rho[i] = rho1
                    p[i] = p1
                    u[i] = -u1

        return x_arr, p, rho, u

#计算sod激波管解析解，源自https://github.com/sbakkerm/Sod-Shock-Tube/tree/main
def analytic_sod(t=0.2):
    """
    计算Sod激波管问题的解析解

    参数:
    t -- 时间(默认为0.2)

    返回:
    data -- 包含x, rho, P, u, e数组的字典
    """
    # 设置初始条件和几何参数
    left_state = (1.0, 1.0, 0.0)  # (P_l, rho_l, u_l)
    right_state = (0.1, 0.125, 0.0)  # (P_r, rho_r, u_r)
    geometry = (-0.5, 0.5, 0.0)  # (x_left, x_right, x_initial)
    gamma = 1.4  # 比热比

    # 使用solve函数计算解析解
    positions, regions, val_dict = solve(
        left_state=left_state,
        right_state=right_state,
        geometry=geometry,
        t=t,
        npts=1000,
        gamma=gamma
    )

    # 提取计算结果
    x = val_dict['x']
    rho = val_dict['rho']
    p = val_dict['p']  # 压力
    u = val_dict['u']  # 速度

    # 计算比内能 e = P / ((γ-1)ρ)
    e = p / ((gamma - 1) * rho)

    # 返回与原始analytic_sod函数相同格式的字典
    return {
        'x': x,
        'rho': rho,
        'P': p,  # 保持大写P表示压力
        'u': u,
        'e': e
    }

#管道特征长度 (x = [-L/2, L/2])

L = 1.0

# 流动参数
Gamma = 1.4  # 比热比
R = 286.9   # 气体常数 (J/kg·K)
Cv = R / (Gamma - 1)            # 定容比热容
Cp = (Gamma * R) / (Gamma - 1)  # 定压比热容

# 网格生成
N = 201   # x方向网格数
xp = np.linspace(-L / 2, L / 2, N)  # 网格点的x坐标
dx = L / (N - 1)  # 网格间距

xp_mid = N // 2  # x=0位置的网格索引(中点)

# 物理量数组初始化
u_arr = np.zeros(N)     # 速度数组
rho_arr = np.zeros(N)   # 密度数组
p_arr = np.zeros(N)     # 压力数组

# 辅助数组
rho = np.zeros(N)
u = np.zeros(N)
p = np.zeros(N)
E = np.zeros(N)
T = np.zeros(N)
c = np.zeros(N)

# 时间步长
dt = 0.001

# 最大计算步数
max_step = 200

# 最大模拟时间
max_tot_time = max_step * dt

#设置初始条件
# 当x < 0时，(左侧: ul, rhol, pl) = (0.0, 1.0, 1.0)
# 当x >= 0时，(右侧: ur, rhor, pr) = (0.0, 0.125, 0.1)
u_arr[0:xp_mid-1] = 0.0
rho_arr[0:xp_mid-1] = 1.0
p_arr[0:xp_mid-1] = 1.0

u_arr[xp_mid-1:] = 0.0
rho_arr[xp_mid-1:] = 0.125
p_arr[xp_mid-1:] = 0.1

#预处理：算法选择设置
#通量分裂方法选择
class FluxSplittingMethod(Enum):
    FVS = 1  # 通量向量分裂
    FDS = 2  # 通量差分裂
#通量向量分裂方法FVS选择
class FVSMethods(Enum):
    STEGER_WARMING = 1   # Steger-Warming方法
    LAX_FRIEDRICH = 2    # Lax-Friedrich方法
    VAN_LEER = 3         # Van Leer方法
#通量差分裂方法FDS选择
class FDSMethods(Enum):
    ROE = 1  # ROE格式
#空间离散格式选择
class SpatialDiscretizationType(Enum):
    SHOCK_CAPTURING = 1  # 激波捕捉格式
#激波捕捉格式选择
class ShockCapturingScheme(Enum):
    TVD_VAN_LEER = 1  # TVD格式(Van Leer限制器)
    WENO = 2          # 5阶WENO格式
    GVC = 3           #群速度控制格式


# 指定通量分裂方法
# 1 - 通量向量分裂(FVS)
# 2 - 通量差分裂(FDS)
flag_flu_spl = FluxSplittingMethod.FDS.value

# FVS方法家族选择
# 1 - Steger-Warming (S-W)
# 2 - Lax-Friedrich (L-F)
# 3 - Van Leer
flag_fvs_met = FVSMethods.VAN_LEER.value

# 指定通量重构方法
# 1 - 直接重构(针对F(U))
# 2 - 特征重构
flag_flu_rec = 1

# FDS方法家族选择(FDS-ROE格式)
flag_fds_met = FDSMethods.ROE.value

# 指定高分辨率通量空间离散格式
flag_spa_typ = SpatialDiscretizationType.SHOCK_CAPTURING.value  # 激波捕捉格式

# 激波捕捉格式类型选择
flag_scs_typ = ShockCapturingScheme.WENO.value

#数组初始化
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

# 计算初始密度、速度、压力、温度、声速和守恒变量
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

# 开始计算和时间步进
# 实际计算时间
t = 0.0
cnt_step = 0

# 主计算循环
while cnt_step < max_step:

    # 更新时间
    t += dt
    cnt_step += 1

    # 时间离散格式选择三阶TVD龙格-库塔法
    if flag_flu_spl == 1:
        # 1 - 通量向量分裂法 (FVS)
        F_p, F_n = Flux_Vect_Split_Common(U, N, Gamma, Cp, Cv, R, flag_fvs_met)
        _, _, _, _, Fx, _, _ = Diff_Cons_Common(N, dx, F_p, F_n, flag_spa_typ, flag_scs_typ)

        # 第一步: U1 = U^n + Δt * Q(U^n)
        U_1 = U + (dt * ((-1) * Fx))

        # 使用U1计算Q(U1)
        F_p_1, F_n_1 = Flux_Vect_Split_Common(U_1, N, Gamma, Cp, Cv, R, flag_fvs_met)
        _, _, _, _, Fx_1, _, _ = Diff_Cons_Common(N, dx, F_p_1, F_n_1, flag_spa_typ, flag_scs_typ)

        # 第二步: U2 = (3/4)U^n + (1/4)U1 + (1/4)Δt * Q(U1)
        U_2 = ((3 / 4) * U) + ((1 / 4) * U_1) + (((1 * dt) / 4) * ((-1) * Fx_1))

        # 使用U2计算Q(U2)
        F_p_2, F_n_2 = Flux_Vect_Split_Common(U_2, N, Gamma, Cp, Cv, R, flag_fvs_met)
        xs_new, xt_new, Fh_p, Fh_n, Fx_2, Fx_p, Fx_n = Diff_Cons_Common(N, dx, F_p_2, F_n_2, flag_spa_typ, flag_scs_typ)

        # 最终更新: U^{n+1} = (1/3)U^n + (2/3)U2 + (2/3)Δt * Q(U2)
        U = ((1 / 3) * U) + ((2 / 3) * U_2) + (((2 / 3) * dt) * ((-1) * Fx_2))

    elif flag_flu_spl == 2:
        xs_new, xt_new, Fx = Flux_Diff_Split_Common(U, N, dx, Gamma, Cp, Cv, R,flag_fds_met, flag_spa_typ,flag_scs_typ)

        # 第一步：计算中间解U_1
        U_1 = U + (dt * (-1 * Fx))

        # 使用U_1再次计算通量
        _, _, Fx_1 = Flux_Diff_Split_Common(U_1, N, dx, Gamma, Cp, Cv, R,flag_fds_met, flag_spa_typ,flag_scs_typ)

        # 第二步：计算中间解U_2
        U_2 = ((3 / 4) * U) + ((1 / 4) * U_1) + ((1 / 4) * dt * (-1 * Fx_1))

        # 使用U_2再次计算通量
        _, _, Fx_2 = Flux_Diff_Split_Common(U_2, N, dx, Gamma, Cp, Cv, R,flag_fds_met, flag_spa_typ,flag_scs_typ)

        # 最终更新：计算新时间步的解
        U = ((1 / 3) * U) + ((2 / 3) * U_2) + ((2 / 3) * dt * (-1 * Fx_2))

# 保存时间t_end时刻的最终解U
U_end = U.copy()
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

#可视化
# 构建方法描述字符串
method_desc = ""
if flag_flu_spl == FluxSplittingMethod.FVS.value:
    fvs_methods = {
        FVSMethods.STEGER_WARMING.value: "Steger-Warming",
        FVSMethods.LAX_FRIEDRICH.value: "Lax-Friedrich",
        FVSMethods.VAN_LEER.value: "Van Leer"
    }
    scs_methods = {
        ShockCapturingScheme.TVD_VAN_LEER.value: "TVD (Van Leer Limiter)",
        ShockCapturingScheme.WENO.value: "5th-order WENO",
        ShockCapturingScheme.GVC.value: "Group Velocity Control (GVC)"
    }
    method_desc = f"FVS-{fvs_methods[flag_fvs_met]} + {scs_methods[flag_scs_typ]}"

elif flag_flu_spl == FluxSplittingMethod.FDS.value:
    fds_methods = {
        FDSMethods.ROE.value: "Roe"
    }

    method_desc = f"FDS-{fds_methods[flag_fds_met]} "
# 创建最终结果图（添加背景色）
h_end = plt.figure(figsize=(10, 8), facecolor='white')

# Subplot 1: 密度
ax1 = plt.subplot(2, 2, 1)
ax1.set_title(f"Density Distribution (t = {t_end:.3f} s)")
ax1.plot(data_end['x'], data_end['rho'], '-b', linewidth=1.5, label='Exact Solution')
ax1.plot(xp, rho_end, 'bo', markersize=3, label='Numerical Solution', alpha=0.7)
ax1.legend(fontsize=9)
ax1.set_xlabel('Position (m)', fontsize=10)
ax1.set_ylabel('Density (kg/m³)', fontsize=10)
ax1.set_xlim(-0.5, 0.5)
ax1.set_ylim(0.0, 1.0)
ax1.grid(True, linestyle='--', alpha=0.5)

# Subplot 2: 压强
ax2 = plt.subplot(2, 2, 2)
ax2.set_title(f"Pressure Distribution (t = {t_end:.3f} s)")
ax2.plot(data_end['x'], data_end['P'], '-g', linewidth=1.5, label='Exact Solution')
ax2.plot(xp, p_end, 'go', markersize=3, label='Numerical Solution', alpha=0.7)
ax2.legend(fontsize=9)
ax2.set_xlabel('Position (m)', fontsize=10)
ax2.set_ylabel('Pressure (Pa)', fontsize=10)
ax2.set_xlim(-0.5, 0.5)
ax2.set_ylim(0.0, 1.0)
ax2.grid(True, linestyle='--', alpha=0.5)

# Subplot 3: 速度
ax3 = plt.subplot(2, 2, 3)
ax3.set_title(f"Velocity Distribution (t = {t_end:.3f} s)")
ax3.plot(data_end['x'], data_end['u'], '-r', linewidth=1.5, label='Exact Solution')
ax3.plot(xp, u_end, 'ro', markersize=3, label='Numerical Solution', alpha=0.7)
ax3.legend(fontsize=9)
ax3.set_xlabel('Position (m)', fontsize=10)
ax3.set_ylabel('Velocity (m/s)', fontsize=10)
ax3.set_xlim(-0.5, 0.5)
ax3.set_ylim(0.0, 1.0)
ax3.grid(True, linestyle='--', alpha=0.5)

# Subplot 4: 内能
ax4 = plt.subplot(2, 2, 4)
ax4.set_title(f"Specific Internal Energy Distribution (t = {t_end:.3f} s)")
ax4.plot(data_end['x'], data_end['e'], '-m', linewidth=1.5, label='Exact Solution')
ax4.plot(xp, e_end, 'mo', markersize=3, label='Numerical Solution', alpha=0.7)
ax4.legend(fontsize=9)
ax4.set_xlabel('Position (m)', fontsize=10)
ax4.set_ylabel('Specific Internal Energy (J/kg)', fontsize=10)
ax4.set_xlim(-0.5, 0.5)
ax4.set_ylim(1.5, 3.0)
ax4.grid(True, linestyle='--', alpha=0.5)

# 主标题
plt.suptitle(f"Sod Shock Tube Simulation ({method_desc}) at t = {t_end:.3f}s",
             fontsize=16, fontweight='bold')
plt.tight_layout(rect=[0, 0, 1, 0.96])
# 生成文件名并保存为PDF
filename = f"Sod_Shock_{method_desc}_t_{t_end:.3f}s".replace(".", "p")
plt.savefig(filename, bbox_inches='tight', dpi=300)
print(f"Results saved as: {filename}")
plt.show()



