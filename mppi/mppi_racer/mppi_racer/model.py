"""Kinematic bicycle 모델 롤아웃 (후보 K개를 한 번에 계산)."""
import numpy as np


def rollout_with_steering_lag(state, steer0, controls, dt, wheelbase, v_max, steer_max,
                              steer_rate_max, understeer=0.0):
    """조향 속도 제한이 있는 롤아웃.

    실제 조향각은 steer0에서 시작해 명령 조향각을 향해 스텝당 최대 steer_rate_max*dt만큼 움직인다
    (시뮬레이터의 조향 서보와 같은 방식).
    controls: (K, T, 2) = [조향각 명령 rad, 가속도 m/s^2]
    반환: 궤적 (K, T+1, 4) = (x, y, yaw, v), 스텝별 실제 조향각 (K, T).
    """
    K, T, _ = controls.shape
    traj = np.empty((K, T + 1, 4))
    traj[:, 0] = state
    target = np.clip(controls[..., 0], -steer_max, steer_max)
    steer_hist = np.empty((K, T))
    max_step = steer_rate_max * dt
    x = np.full(K, state[0], dtype=float)
    y = np.full(K, state[1], dtype=float)
    yaw = np.full(K, state[2], dtype=float)
    v = np.full(K, state[3], dtype=float)
    steer = np.full(K, steer0, dtype=float)
    for t in range(T):
        steer = steer + np.clip(target[:, t] - steer, -max_step, max_step)
        steer_hist[:, t] = steer
        x = x + v * np.cos(yaw) * dt
        y = y + v * np.sin(yaw) * dt
        if understeer:
            # 고속에서 실제 요레이트가 운동학 값보다 작은 것을 속도 비례로 보정한다(기본 0 = 순수 운동학).
            yaw = yaw + np.maximum(1.0 - understeer * v, 0.5) * v * np.tan(steer) / wheelbase * dt
        else:
            yaw = yaw + v * np.tan(steer) / wheelbase * dt
        v = np.clip(v + controls[:, t, 1] * dt, 0.0, v_max)
        traj[:, t + 1, 0] = x
        traj[:, t + 1, 1] = y
        traj[:, t + 1, 2] = yaw
        traj[:, t + 1, 3] = v
    return traj, steer_hist


def rollout(state, controls, dt, wheelbase, v_max, steer_max):
    """조향이 즉시 따라온다고 보는 롤아웃. 반환: (K, T+1, 4)."""
    return rollout_with_steering_lag(state, 0.0, controls, dt, wheelbase, v_max, steer_max,
                                     steer_rate_max=np.inf)[0]


# gym(f110_gym) 단일 트랙(ST) 모델 파라미터 (stack_master/config/SIM/sim_params.yaml)
ST_PARAMS = dict(mu=1.0489, c_sf=4.718, c_sr=5.4562, lf=0.15875, lr=0.17145, h=0.074, m=3.47, inertia=0.04712,
                 v_switch=7.319, a_max=9.51)


def rollout_st(state, steer0, controls, dt, v_max, steer_max, steer_rate_max, substeps=1, params=None,
               yaw_rate0=None, beta0=None):
    """gym과 같은 단일 트랙(타이어) 모델의 롤아웃 (오일러, 스텝당 substeps 서브스텝).

    state=(x, y, yaw, v)는 무게중심 기준이다(gym의 자세와 같음). 초기 요레이트·슬립각은 모르므로
    운동학 정상 상태(yaw_rate=v*tan(steer)/L, beta=atan(lr*tan(steer)/L))로 둔다.
    v<0.5 에서는 gym처럼 운동학으로 전환한다. controls: (K, T, 2) = [조향각 명령, 가속도].
    반환: 궤적 (K, T+1, 4)=(x, y, yaw, v)와 스텝별 조향각 (K, T), rollout_with_steering_lag와 같은 형식.
    """
    P = dict(ST_PARAMS); P.update(params or {})
    mu, csf, csr, lf, lr, h, m, inertia = (P[k] for k in ('mu', 'c_sf', 'c_sr', 'lf', 'lr', 'h', 'm', 'inertia'))
    g = 9.81
    lwb = lf + lr
    K, T, _ = controls.shape
    traj = np.empty((K, T + 1, 4)); traj[:, 0] = state
    target = np.clip(controls[..., 0], -steer_max, steer_max)
    steer_hist = np.empty((K, T))
    x = np.full(K, float(state[0])); y = np.full(K, float(state[1]))
    psi = np.full(K, float(state[2])); v = np.full(K, float(state[3]))
    delta = np.full(K, float(steer0))
    psid = np.full(K, (state[3] * np.tan(steer0) / lwb) if yaw_rate0 is None else float(yaw_rate0))
    beta = np.full(K, np.arctan(lr * np.tan(steer0) / lwb) if beta0 is None else float(beta0))
    hs = dt / substeps
    for t in range(T):
        a_cmd = np.clip(controls[:, t, 1], -P['a_max'], P['a_max'])
        for _ in range(substeps):
            sv = np.clip((target[:, t] - delta) / hs, -steer_rate_max, steer_rate_max)
            pos_lim = np.where(v > P['v_switch'], P['a_max'] * P['v_switch'] / np.maximum(v, 1e-3), P['a_max'])
            a = np.minimum(a_cmd, pos_lim)
            fast = v >= 0.5
            vs = np.maximum(v, 0.5)
            dx = v * np.cos(np.where(fast, beta, 0.0) + psi)
            dy = v * np.sin(np.where(fast, beta, 0.0) + psi)
            d_psid_st = (-mu * m / (vs * inertia * lwb) * (lf ** 2 * csf * (g * lr - a * h) + lr ** 2 * csr * (g * lf + a * h)) * psid
                         + mu * m / (inertia * lwb) * (lr * csr * (g * lf + a * h) - lf * csf * (g * lr - a * h)) * beta
                         + mu * m / (inertia * lwb) * lf * csf * (g * lr - a * h) * delta)
            d_beta_st = ((mu / (vs ** 2 * lwb) * (csr * (g * lf + a * h) * lr - csf * (g * lr - a * h) * lf) - 1.0) * psid
                         - mu / (vs * lwb) * (csr * (g * lf + a * h) + csf * (g * lr - a * h)) * beta
                         + mu / (vs * lwb) * (csf * (g * lr - a * h)) * delta)
            d_psi = np.where(fast, psid, v / lwb * np.tan(delta))
            d_psid_ks = a / lwb * np.tan(delta) + v / (lwb * np.cos(delta) ** 2) * sv
            x = x + hs * dx; y = y + hs * dy
            psi = psi + hs * d_psi
            psid = psid + hs * np.where(fast, d_psid_st, d_psid_ks)
            beta = beta + hs * np.where(fast, d_beta_st, 0.0)
            delta = np.clip(delta + hs * sv, -steer_max, steer_max)
            v = np.clip(v + hs * a, 0.0, v_max)
        steer_hist[:, t] = delta
        traj[:, t + 1, 0] = x; traj[:, t + 1, 1] = y; traj[:, t + 1, 2] = psi; traj[:, t + 1, 3] = v
    return traj, steer_hist


try:  # numba가 있으면 ST 롤아웃을 컴파일해 쓴다(없으면 위 numpy 구현)
    from numba import njit

    @njit(cache=True)
    def _st_kernel(x0, y0, psi0, v0, delta0, psid0, beta0, target, acc, dt, substeps, steer_rate_max, steer_max,
                   v_max, mu, csf, csr, lf, lr, h, m, inertia, v_switch, a_max, traj, steer_hist):
        K, T = target.shape
        g = 9.81
        lwb = lf + lr
        hs = dt / substeps
        for k in range(K):
            x = x0; y = y0; psi = psi0; v = v0; delta = delta0; psid = psid0; beta = beta0
            traj[k, 0, 0] = x; traj[k, 0, 1] = y; traj[k, 0, 2] = psi; traj[k, 0, 3] = v
            for t in range(T):
                a_cmd = acc[k, t]
                if a_cmd > a_max:
                    a_cmd = a_max
                elif a_cmd < -a_max:
                    a_cmd = -a_max
                tg = target[k, t]
                for _ in range(substeps):
                    sv = (tg - delta) / hs
                    if sv > steer_rate_max:
                        sv = steer_rate_max
                    elif sv < -steer_rate_max:
                        sv = -steer_rate_max
                    a = a_cmd
                    if v > v_switch:
                        lim = a_max * v_switch / v
                        if a > lim:
                            a = lim
                    if v >= 0.5:
                        af = g * lr - a * h
                        ar = g * lf + a * h
                        dxx = v * np.cos(beta + psi)
                        dyy = v * np.sin(beta + psi)
                        dpsi = psid
                        dpsid = (-mu * m / (v * inertia * lwb) * (lf * lf * csf * af + lr * lr * csr * ar) * psid
                                 + mu * m / (inertia * lwb) * (lr * csr * ar - lf * csf * af) * beta
                                 + mu * m / (inertia * lwb) * lf * csf * af * delta)
                        dbeta = ((mu / (v * v * lwb) * (csr * ar * lr - csf * af * lf) - 1.0) * psid
                                 - mu / (v * lwb) * (csr * ar + csf * af) * beta
                                 + mu / (v * lwb) * (csf * af) * delta)
                    else:
                        dxx = v * np.cos(psi)
                        dyy = v * np.sin(psi)
                        dpsi = v / lwb * np.tan(delta)
                        dpsid = a / lwb * np.tan(delta) + v / (lwb * np.cos(delta) ** 2) * sv
                        dbeta = 0.0
                    x += hs * dxx
                    y += hs * dyy
                    psi += hs * dpsi
                    psid += hs * dpsid
                    beta += hs * dbeta
                    delta += hs * sv
                    if delta > steer_max:
                        delta = steer_max
                    elif delta < -steer_max:
                        delta = -steer_max
                    v += hs * a
                    if v < 0.0:
                        v = 0.0
                    elif v > v_max:
                        v = v_max
                steer_hist[k, t] = delta
                traj[k, t + 1, 0] = x; traj[k, t + 1, 1] = y; traj[k, t + 1, 2] = psi; traj[k, t + 1, 3] = v

    def rollout_st_fast(state, steer0, controls, dt, v_max, steer_max, steer_rate_max, substeps=2, params=None,
                        yaw_rate0=None, beta0=None):
        P = dict(ST_PARAMS); P.update(params or {})
        K, T, _ = controls.shape
        lwb = P['lf'] + P['lr']
        target = np.ascontiguousarray(np.clip(controls[..., 0], -steer_max, steer_max))
        acc = np.ascontiguousarray(controls[..., 1])
        psid0 = (state[3] * np.tan(steer0) / lwb) if yaw_rate0 is None else float(yaw_rate0)
        beta_0 = np.arctan(P['lr'] * np.tan(steer0) / lwb) if beta0 is None else float(beta0)
        traj = np.empty((K, T + 1, 4)); steer_hist = np.empty((K, T))
        _st_kernel(float(state[0]), float(state[1]), float(state[2]), float(state[3]), float(steer0), float(psid0),
                   float(beta_0), target, acc, float(dt), int(substeps), float(steer_rate_max), float(steer_max),
                   float(v_max), P['mu'], P['c_sf'], P['c_sr'], P['lf'], P['lr'], P['h'], P['m'], P['inertia'],
                   P['v_switch'], P['a_max'], traj, steer_hist)
        return traj, steer_hist
except Exception:  # pragma: no cover
    rollout_st_fast = None
