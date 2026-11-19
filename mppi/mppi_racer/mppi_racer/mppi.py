"""MPPI 통합 플래너·제어기 본체 (ROS 비의존)."""
from dataclasses import dataclass
from time import perf_counter

import numpy as np

from . import costs
from .model import rollout_with_steering_lag, rollout_st, rollout_st_fast


@dataclass
class MPPIParams:
    # 샘플링
    horizon: int = 30            # 스텝 수 (horizon * dt = 1.5 s)
    dt: float = 0.05
    samples: int = 1024
    lam: float = 1.0             # 온도
    sigma_steer: float = 0.12    # rad
    sigma_accel: float = 2.5     # m/s^2
    noise_knots: int = 6         # 잡음 매듭점 수 (호라이즌 전체에 균등 배치)
    exploration_fraction: float = 0.0  # opt-in: candidates independent of the warm start
    # 차량
    wheelbase: float = 0.33
    steer_max: float = 0.4189
    steer_rate_max: float = 3.2  # rad/s (시뮬레이터 sv_max)
    model_st_substeps: int = 0   # 1 이상이면 gym ST 모델 롤아웃(스텝당 서브스텝 수, 2 이상 권장). 0이면 기존 운동학 모델
    yaw_understeer: float = 0.0  # 요레이트 이득 = max(1 - 이 값 * v, 0.5). 0이면 순수 운동학(기존 동작)
    v_max: float = 10.0
    a_max: float = 8.0
    half_width: float = 0.155
    half_length: float = 0.29
    # 비용 가중치
    w_d: float = 10.0
    w_heading: float = 2.0
    w_speed: float = 1.0
    w_speed_under: float = -1.0  # 0 이상이면 v가 목표보다 낮을 때의 속도 가중치(비대칭). -1이면 대칭(기존 동작)
    vref_gain: float = 1.0       # 속도 항의 목표 속도 배율(1.0 = 기존 동작, a_lat 항은 영향 없음)
    a_lat_max: float = 6.0
    w_alat: float = 5.0
    w_alat_car: float = 20.0     # 차량 자체 횡가속도(v^2·tan(steer)/L) 한계 초과
    w_bounds: float = 500.0
    wall_hit_w: float = 0.0       # >0이면 차체가 경계를 실제로 넘는 롤아웃 스텝에 고정 페널티(0=기존 동작)
    bounds_margin: float = 0.10
    obs_margin_s: float = 0.40
    obs_margin_d: float = 0.25
    w_collision: float = 1000.0
    w_near: float = 30.0
    ttc_threshold: float = 1.5
    w_ttc: float = 20.0
    w_terminal: float = 0.0      # 호라이즌 마지막 스텝의 추종 비용(d, 헤딩)에 곱할 배수. 0이면 기존 동작
    # 순수 추적(PP) 조향 열을 명목 입력으로 섞는다. 0이면 기존 동작.
    pp_fraction: float = 0.0     # 후보 중 PP 조향 열 + 잡음으로 대체할 비율
    pp_ld_min: float = 0.8       # 전방주시거리 하한 (m)
    pp_ld_gain: float = 0.25     # 전방주시거리 = max(pp_ld_min, pp_ld_gain * v)
    # 장애물 회피용 사전분포(손으로 만든 후보라 pp_fraction과 같은 성격). 0이면 기존 동작.
    offset_fraction: float = 0.0   # 후보 중 경로 오프셋 따라가기 조향 열로 대체할 비율
    offset_max: float = 1.0        # 최대 횡 오프셋 (m), -max..+max를 offset_n개로 나눔
    offset_n: int = 6
    offset_only_obstacles: int = 1  # 1이면 장애물이 보일 때만 사용
    # 종방향 분리: 1이면 속도 명령을 기준 속도·곡률 제한 프로파일로 직접 낸다(MPPI는 조향 중심). 0이면 기존 동작.
    speed_profile_mode: int = 0
    profile_a_lat: float = 6.0   # 곡률 제한 속도 sqrt(a_lat/|kappa|)
    profile_a_dec: float = 4.0   # 앞 구간 제한 속도에 맞춰 줄이는 감속도 (m/s^2)
    profile_dist: float = 12.0   # 내다보는 거리 (m)
    profile_brake_thr: float = 1.0  # 모드 2: MPPI 계획 가속도가 -이 값보다 낮으면 계획 속도와 min
    profile_a_acc: float = 4.0  # 프로파일 속도 명령으로 허용하는 가속도 (m/s^2, 명령 = v + a/k_accel 이하)
    side_pref_w: float = 0.0       # 통과 쪽 선호 비용 가중치(0=기존 동작). 장애물이 앞 side_pref_window m 안이면 여유 큰 쪽으로 끌고 w_d를 그만큼 줄임
    side_pref_window: float = 6.0  # m
    side_pref_extra: float = 0.15  # m, 목표 d = 장애물 가장자리 ± (half_width + obs_margin_d + extra)
    side_pref_clip: int = 0        # 1이면 통과 목표 d를 각 스텝에서 차가 있는 위치의 트랙 경계(차폭+bounds_margin 안쪽)로 제한(0=기존 동작)
    side_keep: int = 0             # 1이면 한 번 고른 통과 쪽을 유지(히스테리시스)하고 인식에서 사라져도 side_keep_ttl 동안 기억(side_pref_w>0 필요). 0=기존 동작
    side_keep_ttl: float = 1.5     # s
    side_hyst: float = 0.3         # m, 기억한 쪽보다 반대쪽 여유가 이만큼 더 커야 바꿈
    obs_fallback: int = 0         # 1이면 장애물이 보일 때 운동학 모델·기본 잡음으로 되돌림(0=기존 동작)
    obs_yaw_understeer: float = 0.05
    obs_sigma_steer: float = 0.12
    obs_sigma_accel: float = 2.5
    w_dsteer_curv: float = 0.0   # 고곡률에서만 조향 변화율 비용을 w_dsteer*(1+이 값*|kappa|)로 올림(0=기존 동작)
    sigma_curv_gain: float = 0.0  # 고곡률에서 조향 잡음을 1/(1+이 값*|kappa|)배로 줄임(0=기존 동작)
    w_dsteer: float = 20.0
    w_daccel: float = 0.05
    # 비상 감속 보호
    ttc_floor: float = 0.5
    shield_speed: float = 0.0
    # /drive speed로 내보낼 계획 속도의 시점 (스텝)
    speed_step: int = 1


@dataclass
class MPPIResult:
    steer: float
    accel: float
    speed: float
    traj: np.ndarray             # (horizon + 1, 4) 최적 입력열의 예측 궤적
    steer_plan: np.ndarray       # (horizon,) 예측한 실제 조향각
    min_ttc: float
    shield: bool
    cost: float                  # 후보 중 최소 비용


def _wrap_angle(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class MPPI:
    def __init__(self, track, params=None, seed=None):
        self.track = track
        self.params = params or MPPIParams()
        if not 0.0 <= self.params.exploration_fraction <= 1.0:
            raise ValueError('exploration_fraction must be in [0, 1]')
        self.rng = np.random.default_rng(seed)
        self.u = np.zeros((self.params.horizon, 2))
        self.prev_u = np.zeros(2)
        self._idx = None
        self._obs_mode = False
        self._side_mem = []          # side_keep 기억: dict(obs=(6,), sign=±1, t=마지막 관측 시각)
        self._side_obs = np.zeros((0, 6))
        self._side_sign = np.zeros(0)
        self._clock = perf_counter
        self._times = np.arange(1, self.params.horizon + 1) * self.params.dt

    def _sample_noise(self, steer_scale=1.0, sigma_steer=None, sigma_accel=None):
        """시간 방향으로 매끄러운 잡음: 매듭점 몇 개에서만 뽑아 선형 보간한다.

        스텝마다 독립인 잡음은 평균하면 상쇄되어 차선 변경 같은 큰 횡이동 후보가 거의 나오지 않는다.
        첫 후보는 잡음 0(직전 계획 그대로)으로 둔다.
        """
        p = self.params
        ss = p.sigma_steer if sigma_steer is None else sigma_steer
        sa = p.sigma_accel if sigma_accel is None else sigma_accel
        knots = self.rng.normal(size=(p.samples, p.noise_knots, 2)) * (ss * steer_scale, sa)
        pos = np.linspace(0.0, p.noise_knots - 1, p.horizon)
        lo = np.minimum(pos.astype(int), p.noise_knots - 2)
        frac = (pos - lo)[None, :, None]
        noise = knots[:, lo] * (1.0 - frac) + knots[:, lo + 1] * frac
        noise[0] = 0.0
        return noise

    def pp_steering_plan(self, state, steer0, idx0, offset=0.0):
        """현재 속도를 유지하고 순수 추적(PP)으로 조향하는 닫힌 루프 롤아웃의 조향 명령 열 (horizon,).

        offset(m)이 0이 아니면 기준 경로를 왼쪽(+)·오른쪽(-)으로 옮긴 경로를 따라가되, 트랙 경계 안쪽
        (차폭 절반 + 0.15 m 여유)으로 제한한다."""
        p, tr = self.params, self.track
        x, y, yaw, v = (float(a) for a in state)
        steer = float(steer0)
        idx = int(idx0)
        out = np.empty(p.horizon)
        max_step = p.steer_rate_max * p.dt
        ld = max(p.pp_ld_min, p.pp_ld_gain * v)
        ahead = max(1, int(round(ld / tr.spacing)))
        for t in range(p.horizon):
            li = (idx + ahead) % tr.n
            tx, ty = tr.x[li], tr.y[li]
            if offset != 0.0:
                off = float(np.clip(offset, -(tr.d_right[li] - p.half_width - 0.15),
                                    tr.d_left[li] - p.half_width - 0.15))
                tx = tx - tr.sin_psi[li] * off
                ty = ty + tr.cos_psi[li] * off
            alpha = _wrap_angle(np.arctan2(ty - y, tx - x) - yaw)
            cmd = float(np.clip(np.arctan2(2.0 * p.wheelbase * np.sin(alpha), ld),
                                -p.steer_max, p.steer_max))
            out[t] = cmd
            steer += float(np.clip(cmd - steer, -max_step, max_step))
            x += v * np.cos(yaw) * p.dt
            y += v * np.sin(yaw) * p.dt
            yaw += v * np.tan(steer) / p.wheelbase * p.dt
            i2, _, _ = tr.project(np.array([x]), np.array([y]), np.array([idx]))
            idx = int(i2[0])
        return out

    def speed_profile(self, idx):
        """현재 위치(idx)에서 허용되는 속도: 앞 profile_dist m 구간의 제한 속도(기준 속도와 곡률 제한의 최솟값)에
        profile_a_dec로 줄일 수 있는 속도를 거리만큼 더해 그중 최솟값을 취한다."""
        p, tr = self.params, self.track
        n = max(1, int(p.profile_dist / tr.spacing))
        ids = (int(idx) + np.arange(n + 1)) % tr.n
        dist = np.arange(n + 1) * tr.spacing
        v_lat = np.sqrt(p.profile_a_lat / np.maximum(np.abs(tr.kappa[ids]), 1e-3))
        v_lim = np.minimum(tr.vx[ids], v_lat)
        return float(np.min(np.sqrt(v_lim ** 2 + 2.0 * p.profile_a_dec * dist)))

    def _frenet(self, traj, idx0):
        """궤적 (K, T+1, 4)의 1..T 스텝을 직전 스텝의 인덱스를 힌트로 삼아 순서대로 투영한다."""
        K, T1, _ = traj.shape
        idx = np.full(K, idx0)
        out_i = np.empty((K, T1 - 1), dtype=int)
        out_s = np.empty((K, T1 - 1))
        out_d = np.empty((K, T1 - 1))
        for t in range(1, T1):
            idx, s, d = self.track.project(traj[:, t, 0], traj[:, t, 1], idx)
            out_i[:, t - 1], out_s[:, t - 1], out_d[:, t - 1] = idx, s, d
        return out_i, out_s, out_d

    def _rollout(self, state, steer_now, controls):
        p = self.params
        if self._obs_mode:
            # 장애물이 보일 때(obs_fallback): 운동학 모델 + 보정(장애물 조건에서 검증된 기존 구성)
            return rollout_with_steering_lag(state, steer_now, controls, p.dt, p.wheelbase,
                                             p.v_max, p.steer_max, p.steer_rate_max,
                                             understeer=p.obs_yaw_understeer)
        if p.model_st_substeps > 0:
            # 단일 트랙(타이어) 모델 롤아웃(무게중심 기준). yaw_understeer는 쓰지 않는다.
            return (rollout_st_fast or rollout_st)(state, steer_now, controls, p.dt, p.v_max, p.steer_max, p.steer_rate_max,
                              substeps=p.model_st_substeps)
        return rollout_with_steering_lag(state, steer_now, controls, p.dt, p.wheelbase,
                                         p.v_max, p.steer_max, p.steer_rate_max,
                                         understeer=p.yaw_understeer)


    def _update_side_memory(self, obstacles):
        """side_keep: 보이는 장애물에 기억된 통과 쪽을 이어 붙이고(히스테리시스), 안 보이는 기억은 ttl 동안 유지한다."""
        p, tr = self.params, self.track
        now = self._clock()
        car_s = self._idx * tr.spacing
        new_mem = []
        used = set()
        if len(obstacles):
            room_l, room_r = costs.side_room(obstacles, tr, p.bounds_margin)
        for k in range(len(obstacles)):
            o = obstacles[k]
            j = next((q for q, mm in enumerate(self._side_mem) if q not in used
                      and abs(tr.s_diff(0.5 * (mm['obs'][0] + mm['obs'][1]), 0.5 * (o[0] + o[1]))) < 0.6
                      and abs(0.5 * (mm['obs'][2] + mm['obs'][3]) - 0.5 * (o[2] + o[3])) < 0.6), None)
            wider = 1.0 if room_l[k] >= room_r[k] else -1.0
            if j is None:
                sign = wider
            else:
                used.add(j)
                sign = self._side_mem[j]['sign']
                adv = (room_l[k] - room_r[k]) * (-sign)   # 반대쪽이 얼마나 더 넓은가
                if adv > p.side_hyst:
                    sign = -sign
            new_mem.append(dict(obs=o.copy(), sign=sign, t=now))
        for q, mm in enumerate(self._side_mem):
            if q in used:
                continue
            passed = tr.s_diff(car_s, mm['obs'][1]) > 2.0
            if now - mm['t'] <= p.side_keep_ttl and not passed:
                new_mem.append(mm)
        self._side_mem = new_mem
        if new_mem:
            self._side_obs = np.array([mm['obs'] for mm in new_mem]).reshape(-1, 6)
            self._side_sign = np.array([mm['sign'] for mm in new_mem])
        else:
            self._side_obs = np.zeros((0, 6))
            self._side_sign = np.zeros(0)

    def _stage_costs(self, traj, steer, idx0, obstacles):
        """(K, T) 스텝별 비용과 TTC."""
        p, tr = self.params, self.track
        idx, s, d = self._frenet(traj, idx0)
        yaw, v = traj[:, 1:, 2], traj[:, 1:, 3]
        heading_err = _wrap_angle(yaw - tr.psi[idx])
        cost = costs.tracking_cost(d, heading_err, p.w_d, p.w_heading)
        if p.w_terminal > 0.0:
            cost[:, -1] += p.w_terminal * cost[:, -1]
        cost += costs.speed_cost(v, tr.vx[idx], tr.kappa[idx], p.w_speed, p.a_lat_max, p.w_alat,
                                 w_under=(p.w_speed_under if p.w_speed_under >= 0.0 else None),
                                 vref_gain=p.vref_gain)
        cost += costs.lateral_accel_cost(traj[:, :-1, 3], steer, p.wheelbase,
                                         p.a_lat_max, p.w_alat_car)
        cost += costs.bounds_cost(d, tr.d_left[idx], tr.d_right[idx],
                                  p.half_width, p.bounds_margin, p.w_bounds)
        if p.wall_hit_w > 0.0:
            cost += costs.wall_hit_cost(d, tr.d_left[idx], tr.d_right[idx],
                                        p.half_width, p.bounds_margin, p.wall_hit_w)
        ttc = np.full(s.shape, np.inf)
        if len(obstacles):
            t = np.broadcast_to(self._times, s.shape)
            ahead, behind, gap_d = costs.obstacle_gaps(s, d, t, obstacles, tr,
                                                       p.half_width, p.half_length)
            cost += costs.obstacle_cost(ahead, behind, gap_d, p.obs_margin_s, p.obs_margin_d,
                                        p.w_collision, p.w_near)
            ttc = costs.time_to_collision(ahead, gap_d, v, obstacles[:, 4], behind=behind)
            if p.side_pref_w > 0.0 and not p.side_keep:
                targets = costs.side_targets(obstacles, tr, p.half_width, p.obs_margin_d,
                                             p.side_pref_extra, p.bounds_margin)
                lo = hi = None
                if p.side_pref_clip:
                    lo = -(tr.d_right[idx] - p.half_width - p.bounds_margin)
                    hi = tr.d_left[idx] - p.half_width - p.bounds_margin
                side, has = costs.side_pref_cost(ahead, behind, d, targets, p.side_pref_window, p.side_pref_w, lo, hi)
                cost += side - np.minimum(p.side_pref_w, p.w_d) * d ** 2 * has
            cost += costs.ttc_cost(ttc, p.ttc_threshold, p.w_ttc)
        if p.side_pref_w > 0.0 and p.side_keep and len(self._side_obs):
            t = np.broadcast_to(self._times, s.shape)
            s_ahead, s_behind, _ = costs.obstacle_gaps(s, d, t, self._side_obs, tr, p.half_width, p.half_length)
            targets = costs.side_targets(self._side_obs, tr, p.half_width, p.obs_margin_d,
                                         p.side_pref_extra, p.bounds_margin, sign=self._side_sign)
            lo = hi = None
            if p.side_pref_clip:
                lo = -(tr.d_right[idx] - p.half_width - p.bounds_margin)
                hi = tr.d_left[idx] - p.half_width - p.bounds_margin
            side, has = costs.side_pref_cost(s_ahead, s_behind, d, targets, p.side_pref_window, p.side_pref_w, lo, hi)
            cost += side - np.minimum(p.side_pref_w, p.w_d) * d ** 2 * has
        return cost, ttc

    def step(self, state, obstacles, shift=True, steer_now=None, profile=None):
        """현재 상태 (x, y, yaw, v)와 장애물 (M, 6)에서 다음 명령을 계산한다.

        shift=False면 직전 계획을 밀지 않는다(제어 주기가 dt보다 짧을 때 사용).
        steer_now는 현재 실제 조향각 추정값이다. 없으면 직전 조향 명령으로 본다.
        profile에 dict를 주면 각 계산 구간의 경과 시간(ms)을 채운다.
        """
        if profile is not None:
            profile.clear()
            started = checkpoint = perf_counter()

        def mark(name):
            if profile is not None:
                nonlocal checkpoint
                now = perf_counter()
                profile[name] = (now - checkpoint) * 1000.0
                checkpoint = now

        p, tr = self.params, self.track
        state = np.asarray(state, dtype=float)
        obstacles = np.asarray(obstacles, dtype=float).reshape(-1, 6)
        if steer_now is None:
            steer_now = self.prev_u[0]
        if self._idx is None:
            self._idx = tr.locate(state[0], state[1])
        i0, _, _ = tr.project(state[:1], state[1:2], np.array([self._idx]))
        self._idx = int(i0[0])

        # warm start: 직전 최적 입력열을 한 칸 민다.
        if shift:
            self.u = np.vstack((self.u[1:], self.u[-1:]))
        mark('prepare_ms')
        if p.side_keep and p.side_pref_w > 0.0:
            self._update_side_memory(obstacles)
        self._obs_mode = bool(p.obs_fallback and len(obstacles))
        kloc = 0.0
        if not self._obs_mode and (p.sigma_curv_gain > 0.0 or p.w_dsteer_curv > 0.0):
            # 앞쪽 2.5 m 안의 최대 곡률(고곡률 구간 판단)
            ids = (self._idx + np.arange(0, 25)) % tr.n
            kloc = float(np.max(np.abs(tr.kappa[ids])))
        if self._obs_mode:
            noise = self._sample_noise(1.0, p.obs_sigma_steer, p.obs_sigma_accel)
        else:
            noise = self._sample_noise(1.0 / (1.0 + p.sigma_curv_gain * kloc))
        cand = self.u + noise
        fresh = min(p.samples - 1, int(p.samples * p.exploration_fraction))
        if fresh > 0:
            # Preserve candidate zero (the current plan). Fresh candidates can
            # escape a saturated braking warm start after the vehicle stops.
            cand[-fresh:] = noise[-fresh:]
        n_pp = min(p.samples - 1, int(p.samples * p.pp_fraction))
        if n_pp > 0:
            # 후보 0(직전 계획)은 보존하고, 마지막 n_pp개는 PP 조향 열 + 조향 잡음으로 둔다.
            pp = self.pp_steering_plan(state, steer_now, self._idx)
            cand[-n_pp:, :, 0] = pp[None, :] + noise[-n_pp:, :, 0]
        n_off = int(p.samples * p.offset_fraction)
        if n_off > 0 and p.offset_n > 0 and (len(obstacles) or not p.offset_only_obstacles):
            # 장애물 회피용 사전분포: 기준 경로를 좌우로 옮긴 경로를 따라가는 조향 열(+ 잡음)을 후보에 넣는다.
            offs = np.linspace(-p.offset_max, p.offset_max, p.offset_n)
            per = max(1, n_off // len(offs))
            for k, o in enumerate(offs):
                lo = 1 + k * per
                hi = min(lo + per, p.samples)
                if lo >= hi:
                    break
                plan = self.pp_steering_plan(state, steer_now, self._idx, offset=float(o))
                cand[lo:hi, :, 0] = plan[None, :] + noise[lo:hi, :, 0]
        cand[..., 0] = np.clip(cand[..., 0], -p.steer_max, p.steer_max)
        cand[..., 1] = np.clip(cand[..., 1], -p.a_max, p.a_max)
        mark('sample_ms')

        traj, steer = self._rollout(state, steer_now, cand)
        mark('rollout_ms')
        stage, _ = self._stage_costs(traj, steer, self._idx, obstacles)
        total = stage.sum(axis=1) + costs.smoothness_cost(cand, self.prev_u, p.w_dsteer * (1.0 + p.w_dsteer_curv * kloc), p.w_daccel)
        mark('cost_ms')
        weights = np.exp(-(total - total.min()) / p.lam)
        weights /= weights.sum()
        self.u = np.tensordot(weights, cand, axes=1)
        self.prev_u = self.u[0].copy()
        mark('update_ms')

        best, best_steer = self._rollout(state, steer_now, self.u[None])
        _, ttc = self._stage_costs(best, best_steer, self._idx, obstacles)
        min_ttc = float(ttc.min())
        speed = costs.shield_speed(float(best[0, p.speed_step, 3]), min_ttc,
                                   p.ttc_floor, p.shield_speed)
        mark('final_ms')
        if profile is not None:
            profile['total_ms'] = (perf_counter() - started) * 1000.0
        return MPPIResult(steer=float(self.u[0, 0]), accel=float(self.u[0, 1]), speed=speed,
                          traj=best[0], steer_plan=best_steer[0], min_ttc=min_ttc, shield=min_ttc < p.ttc_floor,
                          cost=float(total.min()))
