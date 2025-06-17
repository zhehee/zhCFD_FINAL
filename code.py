import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.optimize import root_scalar
import os
import time

# 全局参数定义
L = 1.0  # 激波管长度
Gamma = 1.4  # 比热比
R = 286.9  # 气体常数 (J/kg·K)
Cv = R / (Gamma - 1)  # 定容比热
Cp = Gamma * R / (Gamma - 1)  # 定压比热

# 网格生成
N = 201  # 网格点数
xp = np.linspace(-L / 2, L / 2, N)  # 网格坐标
dx = L / (N - 1)
xp_mid = N // 2  # 中点位置

# 算法选择标志
flag_flu_spl = 1  # 1=FVS, 2=FDS
flag_fvs_met = 1  # 1=Steger-Warming, 2=Lax-Friedrich, 3=Van Leer, 4=AUSM
flag_flu_rec = 1  # 通量重建方法
flag_spa_typ = 1  # 1=UPW, 2=SCS
flag_upw_typ = 1  # 1=1阶, 2=2阶, 3=3阶, 4=5阶
flag_scs_typ = 1  # 1=TVD, 2=NND, 3=WENO
flag_tim_mar = 4  # 1=Euler, 2=Trapezoid, 3=RK2, 4=RK3TVD, 5=RK4

# 时间步长和计算步
dt = 0.001
max_step = 100
max_tot_time = max_step * dt

# 输出设置
savefolder = f"Program_Sod_Shock_Tube_MaxTime_{max_tot_time:.3f}"
os.makedirs(savefolder, exist_ok=True)
flag_exp_avi = 0
flag_exp_gif = 1


# Sod激波管问题的精确解
def analytic_sod(t):
    # 状态参数：左(1)和右(2)状态
    rho1, u1, p1 = 1.0, 0.0, 1.0
    rho2, u2, p2 = 0.125, 0.0, 0.1

    # 初始音速
    c1 = np.sqrt(Gamma * p1 / rho1)
    c2 = np.sqrt(Gamma * p2 / rho2)

    # 压力比
    p21 = p2 / p1

    # 计算压力比函数
    def pressure_ratio(p):
        Gm = Gamma - 1
        return (p - 1) * np.sqrt(
            (Gm / (2 * Gamma)) * (Gm * (1 + (Gamma + 1) / (Gamma - 1) * p)) / (1 + Gm / Gamma * (1 + p))) - \
            c1 / c2 * Gm * (1 - p21 ** ((Gamma - 1) / (2 * Gamma)))

    # 求解压力比
    sol = root_scalar(pressure_ratio, bracket=[1e-5, 1.0], method='brentq')
    p3p1 = sol.root
    p3 = p3p1 * p1

    # 计算激波速度
    u3 = c1 * Gamma / (Gamma - 1) * (p3p1 - 1) / np.sqrt(0.5 * Gamma * (Gamma - 1) * (1 + Gamma * p3p1))
    rho3 = rho1 * (1 + Gamma * p3p1) / (Gamma + Gamma * p3p1)
    c3 = np.sqrt(Gamma * p3 / rho3)

    # 特征线位置
    x_shock = u3 * t
    x_contact = u3 * t
    x_head = -c1 * t
    x_tail = (u3 - c3) * t

    # 创建结果数组
    n_points = 500
    x = np.linspace(-L / 2, L / 2, n_points)
    rho = np.zeros(n_points)
    u = np.zeros(n_points)
    p = np.zeros(n_points)
    e = np.zeros(n_points)

    # 根据位置计算状态
    for i in range(n_points):
        if x[i] <= x_head:  # 左未扰动区
            rho[i] = rho1
            u[i] = u1
            p[i] = p1
        elif x[i] <= x_tail:  # 稀疏波膨胀区
            c = (x_head - x[i]) / t + c1
            u[i] = 2 / (Gamma + 1) * (c1 + (x[i] / t))
            c = c1 - (Gamma - 1) / 2 * u[i]
            rho[i] = rho1 * (c / c1) ** (2 / (Gamma - 1))
            p[i] = p1 * (rho[i] / rho1) ** Gamma
        elif x[i] <= x_contact:  # 接触面左
            rho[i] = rho3
            u[i] = u3
            p[i] = p3
        elif x[i] <= x_shock:  # 接触面右
            rho[i] = rho2 * ((Gamma + 1) * p3 + (Gamma - 1) * p2) / ((Gamma - 1) * p3 + (Gamma + 1) * p2)
            u[i] = u3
            p[i] = p3
        else:  # 右未扰动区
            rho[i] = rho2
            u[i] = u2
            p[i] = p2

        # 计算内能
        e[i] = p[i] / ((Gamma - 1) * rho[i])

    return {'x': x, 'rho': rho, 'u': u, 'p': p, 'e': e}


# 通量矢分裂 (FVS) 计算
def flux_vect_split_common(U, gamma, method):
    N = U.shape[0]
    F_p = np.zeros((N, 3))
    F_n = np.zeros((N, 3))

    # 计算原始变量
    rho = U[:, 0]
    u = U[:, 1] / rho
    E = U[:, 2]
    p = (gamma - 1) * (E - 0.5 * rho * u ** 2)
    H = (E + p) / rho  # 总焓

    if method == 1:  # Steger-Warming分裂
        for i in range(N):
            # 特征值
            c = np.sqrt(gamma * p[i] / rho[i])
            lambda1 = u[i] - c
            lambda2 = u[i]
            lambda3 = u[i] + c

            # 分裂特征值
            lambda1_p = 0.5 * (lambda1 + np.abs(lambda1))
            lambda1_n = 0.5 * (lambda1 - np.abs(lambda1))
            lambda2_p = 0.5 * (lambda2 + np.abs(lambda2))
            lambda2_n = 0.5 * (lambda2 - np.abs(lambda2))
            lambda3_p = 0.5 * (lambda3 + np.abs(lambda3))
            lambda3_n = 0.5 * (lambda3 - np.abs(lambda3))

            # 正通量
            F_p[i, 0] = (lambda1_p * (gamma - 1) / gamma + lambda2_p + lambda3_p / (gamma)) * rho[i] / (2 * gamma)
            F_p[i, 1] = (lambda1_p * (u[i] - c) * (gamma - 1) / gamma +
                         lambda2_p * u[i] +
                         lambda3_p * (u[i] + c) / gamma) * rho[i] / (2 * gamma)
            F_p[i, 2] = (lambda1_p * (H[i] - u[i] * c) * (gamma - 1) / gamma +
                         lambda2_p * u[i] ** 2 +
                         lambda3_p * (H[i] + u[i] * c) / gamma) * rho[i] / (2 * gamma)

            # 负通量
            F_n[i, 0] = (lambda1_n * (gamma - 1) / gamma + lambda2_n + lambda3_n / gamma) * rho[i] / (2 * gamma)
            F_n[i, 1] = (lambda1_n * (u[i] - c) * (gamma - 1) / gamma +
                         lambda2_n * u[i] +
                         lambda3_n * (u[i] + c) / gamma) * rho[i] / (2 * gamma)
            F_n[i, 2] = (lambda1_n * (H[i] - u[i] * c) * (gamma - 1) / gamma +
                         lambda2_n * u[i] ** 2 +
                         lambda3_n * (H[i] + u[i] * c) / gamma) * rho[i] / (2 * gamma)

    elif method == 2:  # Lax-Friedrichs分裂 (简单实现)
        F = np.zeros((N, 3))
        F[:, 0] = rho * u
        F[:, 1] = rho * u ** 2 + p
        F[:, 2] = u * (E + p)

        alpha = np.max(np.abs(u) + np.sqrt(gamma * p / rho))  # 最大特征值

        for i in range(N):
            F_p[i] = 0.5 * (F[i] + alpha * U[i])
            F_n[i] = 0.5 * (F[i] - alpha * U[i])

    return F_p, F_n


# Roe方法 (通量差分裂)
def flux_diff_split_roe(U, dx, gamma):
    N = U.shape[0]
    F_roe = np.zeros((N - 1, 3))  # 网格边界处的通量

    # 网格单元变量
    rho = U[:, 0]
    u = U[:, 1] / rho
    E = U[:, 2]
    p = (gamma - 1) * (E - 0.5 * rho * u ** 2)
    H = (E + p) / rho

    # 计算单元边界处的通量
    for j in range(1, N - 1):  # j是网格单元的索引
        # 左右状态 (j, j+1)
        rhoL, uL, pL, HL = rho[j], u[j], p[j], H[j]
        rhoR, uR, pR, HR = rho[j + 1], u[j + 1], p[j + 1], H[j + 1]

        # Roe平均
        sqrt_rhoL = np.sqrt(rhoL)
        sqrt_rhoR = np.sqrt(rhoR)
        u_roe = (sqrt_rhoL * uL + sqrt_rhoR * uR) / (sqrt_rhoL + sqrt_rhoR)
        H_roe = (sqrt_rhoL * HL + sqrt_rhoR * HR) / (sqrt_rhoL + sqrt_rhoR)
        c_roe = np.sqrt((gamma - 1) * (H_roe - 0.5 * u_roe ** 2))

        # 左右通量
        F_L = np.array([rhoL * uL,
                        rhoL * uL ** 2 + pL,
                        uL * (E[j] + pL)])
        F_R = np.array([rhoR * uR,
                        rhoR * uR ** 2 + pR,
                        uR * (E[j + 1] + pR)])

        # 特征值分解
        delta_u = uR - uL
        delta_p = pR - pL
        delta_rho = rhoR - rhoL

        # Roe波强度
        delta_w = np.zeros(3)
        delta_w[0] = 0.5 * (delta_p - rho_roe * c_roe * delta_u) / c_roe ** 2
        delta_w[1] = delta_rho - delta_p / c_roe ** 2
        delta_w[2] = 0.5 * (delta_p + rho_roe * c_roe * delta_u) / c_roe ** 2

        # 特征值
        lambda1 = np.abs(u_roe - c_roe)
        lambda2 = np.abs(u_roe)
        lambda3 = np.abs(u_roe + c_roe)

        # 计算|A|ΔU
        A_dU = np.array([
            lambda1 * delta_w[0] + lambda2 * delta_w[1] + lambda3 * delta_w[2],
            lambda1 * (u_roe - c_roe) * delta_w[0] + lambda2 * u_roe * delta_w[1] + lambda3 * (u_roe + c_roe) * delta_w[
                2],
            lambda1 * (H_roe - u_roe * c_roe) * delta_w[0] + lambda2 * (0.5 * u_roe ** 2) * delta_w[1] + lambda3 * (
                        H_roe + u_roe * c_roe) * delta_w[2]
        ])

        # Roe通量
        F_roe[j] = 0.5 * (F_L + F_R) - 0.5 * A_dU

    return F_roe


# 空间导数计算 (通用)
def diff_cons_common(N, dx, F_p, F_n, spa_type, upw_type, scs_type):
    Fx = np.zeros((N, 3))
    Fh_p = np.zeros((N - 1, 3))  # 边界处正通量
    Fh_n = np.zeros((N - 1, 3))  # 边界处负通量

    # 计算网格单元边界处的数值通量
    if spa_type == 1:  # UPW迎风
        for j in range(0, N - 1):
            if upw_type == 1:  # 一阶迎风
                # F_{j+1/2} = F^+_j + F^-_{j+1}
                Fh_p[j] = F_p[j]
                Fh_n[j] = F_n[j + 1]
            elif upw_type == 2:  # 二阶迎风 (简化为Lax-Wendroff)
                # 中心差分校正
                Fh_p[j] = 0.5 * (F_p[j] + F_p[j + 1])
                Fh_n[j] = 0.5 * (F_n[j + 1] + F_n[j])

    # 计算通量梯度 (F_x)
    for i in range(1, N - 1):
        Fx[i] = (Fh_p[i] + Fh_n[i] - Fh_p[i - 1] - Fh_n[i - 1]) / dx

    # 边界条件 (零梯度)
    Fx[0] = Fx[1]
    Fx[N - 1] = Fx[N - 2]

    return Fh_p, Fh_n, Fx


# 可视化函数
def plot_props(t, x, rho, p, u, e):
    plt.figure(figsize=(16, 12))

    plt.subplot(2, 2, 1)
    plt.plot(x, rho, 'b-', linewidth=2)
    plt.title(f'Density at t={t:.4f}')
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(x, p, 'r-', linewidth=2)
    plt.title(f'Pressure at t={t:.4f}')
    plt.xlabel('x')
    plt.ylabel('Pressure')
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(x, u, 'g-', linewidth=2)
    plt.title(f'Velocity at t={t:.4f}')
    plt.xlabel('x')
    plt.ylabel('Velocity')
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(x, e, 'm-', linewidth=2)
    plt.title(f'Specific Internal Energy at t={t:.4f}')
    plt.xlabel('x')
    plt.ylabel('Internal Energy')
    plt.grid(True)

    plt.tight_layout()
    plt.draw()
    plt.pause(0.01)


# 主计算函数
def main():
    global xp, L, Gamma, R, Cv, Cp, dt, max_step, flag_flu_spl, flag_fvs_met, flag_spa_typ, flag_upw_typ

    # 初始条件设置
    u_arr = np.zeros(N)
    rho_arr = np.zeros(N)
    p_arr = np.zeros(N)

    # 左半部 (x < 0): 高压区
    u_arr[:xp_mid] = 0.0
    rho_arr[:xp_mid] = 1.0
    p_arr[:xp_mid] = 1.0

    # 右半部 (x >= 0): 低压区
    u_arr[xp_mid:] = 0.0
    rho_arr[xp_mid:] = 0.125
    p_arr[xp_mid:] = 0.1

    # 守恒变量初始化
    U = np.zeros((N, 3))
    E_arr = p_arr / ((Gamma - 1) * rho_arr) + 0.5 * u_arr ** 2
    U[:, 0] = rho_arr  # 质量
    U[:, 1] = rho_arr * u_arr  # 动量
    U[:, 2] = rho_arr * E_arr  # 总能量

    # 时间步进
    t = 0.0
    cnt_step = 0

    # 用于存储动画帧
    frames = []

    # 主计算循环
    while cnt_step < max_step:
        t += dt
        cnt_step += 1

        # Euler 时间推进
        if flag_tim_mar == 1:
            if flag_flu_spl == 1:  # FVS
                F_p, F_n = flux_vect_split_common(U, Gamma, flag_fvs_met)
                Fh_p, Fh_n, Fx = diff_cons_common(N, dx, F_p, F_n, flag_spa_typ, flag_upw_typ, flag_scs_typ)
                U += dt * (-Fx)  # 更新守恒变量

        # 3阶TVD Runge-Kutta 时间推进
        elif flag_tim_mar == 4:
            # 第一阶段
            F_p, F_n = flux_vect_split_common(U, Gamma, flag_fvs_met)
            _, _, Fx = diff_cons_common(N, dx, F_p, F_n, flag_spa_typ, flag_upw_typ, flag_scs_typ)
            U1 = U + dt * (-Fx)

            # 第二阶段
            F_p1, F_n1 = flux_vect_split_common(U1, Gamma, flag_fvs_met)
            _, _, Fx1 = diff_cons_common(N, dx, F_p1, F_n1, flag_spa_typ, flag_upw_typ, flag_scs_typ)
            U2 = (0.75 * U) + (0.25 * U1) + (0.25 * dt * (-Fx1))

            # 第三阶段
            F_p2, F_n2 = flux_vect_split_common(U2, Gamma, flag_fvs_met)
            Fh_p, Fh_n, Fx2 = diff_cons_common(N, dx, F_p2, F_n2, flag_spa_typ, flag_upw_typ, flag_scs_typ)
            U = (1 / 3 * U) + (2 / 3 * U2) + (2 / 3 * dt * (-Fx2))

        # 提取物理量
        rho = U[:, 0]
        u_vel = U[:, 1] / rho
        e_int = U[:, 2] / rho - 0.5 * u_vel ** 2
        p = (Gamma - 1) * rho * e_int

        # 保存当前帧用于动画
        frames.append((t, rho.copy(), p.copy(), u_vel.copy(), e_int.copy()))

        # 每10步输出一次进度
        if cnt_step % 10 == 0:
            print(f"Step: {cnt_step}/{max_step}, Time: {t:.4f} s, Max density: {np.max(rho):.4f}")

    # 最终时间点结果
    rho_end = U[:, 0]
    u_end = U[:, 1] / rho_end
    e_end = U[:, 2] / rho_end - 0.5 * u_end ** 2
    p_end = (Gamma - 1) * rho_end * e_end

    # 获取精确解
    exact = analytic_sod(t)

    # 绘制对比图
    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    plt.plot(exact['x'], exact['rho'], 'b-', label='Exact', linewidth=2)
    plt.plot(xp, rho_end, 'ro', markersize=3, label='Numerical')
    plt.title('Density Comparison')
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.plot(exact['x'], exact['p'], 'g-', label='Exact', linewidth=2)
    plt.plot(xp, p_end, 'go', markersize=3, label='Numerical')
    plt.title('Pressure Comparison')
    plt.xlabel('x')
    plt.ylabel('Pressure')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(exact['x'], exact['u'], 'r-', label='Exact', linewidth=2)
    plt.plot(xp, u_end, 'ro', markersize=3, label='Numerical')
    plt.title('Velocity Comparison')
    plt.xlabel('x')
    plt.ylabel('Velocity')
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(exact['x'], exact['e'], 'm-', label='Exact', linewidth=2)
    plt.plot(xp, e_end, 'mo', markersize=3, label='Numerical')
    plt.title('Specific Internal Energy Comparison')
    plt.xlabel('x')
    plt.ylabel('Internal Energy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'{savefolder}/final_comparison.png')

    # 创建动画
    if flag_exp_gif:
        fig, axs = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Sod Shock Tube Simulation')

        def update(frame_num):
            t, rho, p, u, e = frames[frame_num]
            fig.suptitle(f'Time: {t:.4f} s')

            axs[0, 0].clear()
            axs[0, 0].plot(xp, rho, 'b-')
            axs[0, 0].set_title('Density')
            axs[0, 0].set_xlabel('x')
            axs[0, 0].set_ylabel('Density')
            axs[0, 0].grid(True)

            axs[0, 1].clear()
            axs[0, 1].plot(xp, p, 'r-')
            axs[0, 1].set_title('Pressure')
            axs[0, 1].set_xlabel('x')
            axs[0, 1].set_ylabel('Pressure')
            axs[0, 1].grid(True)

            axs[1, 0].clear()
            axs[1, 0].plot(xp, u, 'g-')
            axs[1, 0].set_title('Velocity')
            axs[1, 0].set_xlabel('x')
            axs[1, 0].set_ylabel('Velocity')
            axs[1, 0].grid(True)

            axs[1, 1].clear()
            axs[1, 1].plot(xp, e, 'm-')
            axs[1, 1].set_title('Specific Internal Energy')
            axs[1, 1].set_xlabel('x')
            axs[1, 1].set_ylabel('Internal Energy')
            axs[1, 1].grid(True)

            return fig,

        ani = animation.FuncAnimation(fig, update, frames=len(frames), interval=50, blit=True)
        ani.save(f'{savefolder}/sod_shock_tube_simulation.gif', writer='pillow', fps=20)

    print("Simulation completed successfully!")


if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Execution time: {time.time() - start_time:.2f} seconds")