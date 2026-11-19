import numpy as np
import pytest

from mppi_racer import costs
from mppi_racer.model import rollout
from mppi_racer.mppi import MPPI, MPPIParams
from mppi_racer.track import Track

R = 10.0
NO_OBS = np.zeros((0, 6))


def make_track(v_ref=3.0, half=1.5):
    n = 400
    th = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    return Track(R * np.cos(th), R * np.sin(th), vx=np.full(n, v_ref),
                 d_left=np.full(n, half), d_right=np.full(n, half))


def state_on_track(s, d=0.0, v=2.0):
    """원형 트랙의 프레넷 (s, d)에 해당하는 상태. +d는 원 안쪽."""
    th = s / R
    return np.array([(R - d) * np.cos(th), (R - d) * np.sin(th), th + np.pi / 2, v])


def simulate(ctrl, state, obstacles, steps):
    """같은 bicycle 모델을 플랜트로 쓴 폐루프 시뮬레이션. (s, d, v, 충돌 여부) 기록을 돌려준다."""
    p, tr = ctrl.params, ctrl.track
    log = []
    idx = tr.locate(state[0], state[1])
    for k in range(steps):
        res = ctrl.step(state, obstacles)
        accel = -p.a_max if res.shield else res.accel
        u = np.array([[[res.steer, accel]]])
        state = rollout(state, u, p.dt, p.wheelbase, p.v_max, p.steer_max)[0, 1]
        idx, s, d = tr.project(state[:1], state[1:2], np.array([idx]))
        idx = int(idx[0])
        ahead, behind, gap_d = costs.obstacle_gaps(s, d, np.zeros(1), obstacles, tr,
                                                   p.half_width, p.half_length)
        hit = bool(np.any((np.maximum(ahead, behind) < 0) & (gap_d < 0)))
        log.append((s[0], d[0], state[3], hit))
    return np.array(log)


def test_step_returns_command_within_limits_and_planned_trajectory():
    p = MPPIParams()
    ctrl = MPPI(make_track(), p, seed=0)
    res = ctrl.step(state_on_track(1.0), NO_OBS)
    assert abs(res.steer) <= p.steer_max
    assert 0.0 <= res.speed <= p.v_max
    assert res.traj.shape == (p.horizon + 1, 4)
    assert res.shield is False and np.isinf(res.min_ttc)


def test_same_seed_gives_same_command():
    a = MPPI(make_track(), MPPIParams(), seed=3).step(state_on_track(1.0, d=0.2), NO_OBS)
    b = MPPI(make_track(), MPPIParams(), seed=3).step(state_on_track(1.0, d=0.2), NO_OBS)
    assert a.steer == b.steer and a.speed == b.speed


def test_converges_to_path_and_reference_speed_without_obstacles():
    ctrl = MPPI(make_track(v_ref=3.0), MPPIParams(), seed=0)
    log = simulate(ctrl, state_on_track(1.0, d=0.4, v=1.0), NO_OBS, steps=200)
    tail = log[100:]
    assert np.abs(tail[:, 1]).max() < 0.10
    assert np.abs(tail[:, 2] - 3.0).max() < 0.4


def test_stays_inside_track_when_starting_near_boundary():
    ctrl = MPPI(make_track(half=1.0), MPPIParams(), seed=1)
    log = simulate(ctrl, state_on_track(1.0, d=-0.7, v=2.0), NO_OBS, steps=120)
    assert np.abs(log[:, 1]).max() < 1.0 - ctrl.params.half_width


@pytest.mark.parametrize('seed', [0, 1, 2])
def test_passes_static_obstacle_on_raceline_without_collision(seed):
    ctrl = MPPI(make_track(), MPPIParams(), seed=seed)
    obs = np.array([[8.0, 8.3, -0.15, 0.15, 0.0, 0.0]])
    log = simulate(ctrl, state_on_track(1.0, v=3.0), obs, steps=160)
    assert not log[:, 3].any()
    assert log[-1, 0] > 9.5                              # 장애물을 지나감
    assert abs(log[-1, 1]) < 0.15                        # 지난 뒤 경로로 복귀


def test_passes_two_staggered_obstacles_without_collision():
    ctrl = MPPI(make_track(), MPPIParams(), seed=0)
    obs = np.array([[7.0, 7.3, -0.6, -0.1, 0.0, 0.0],
                    [10.0, 10.3, 0.1, 0.6, 0.0, 0.0]])
    log = simulate(ctrl, state_on_track(1.0, v=3.0), obs, steps=220)
    assert not log[:, 3].any()
    assert log[-1, 0] > 12.0


def test_stops_before_obstacle_blocking_the_whole_track():
    ctrl = MPPI(make_track(), MPPIParams(), seed=0)
    obs = np.array([[8.0, 8.3, -1.5, 1.5, 0.0, 0.0]])
    log = simulate(ctrl, state_on_track(1.0, v=3.0), obs, steps=200)
    assert not log[:, 3].any()
    assert log[-1, 2] < 0.3
    assert log[-1, 0] < 8.0


def test_follows_slower_moving_obstacle_blocking_the_track_without_collision():
    ctrl = MPPI(make_track(), MPPIParams(), seed=0)
    p, tr = ctrl.params, ctrl.track
    state = state_on_track(1.0, v=3.0)
    obs = np.array([[5.0, 5.3, -1.5, 1.5, 1.0, 0.0]])
    hits = []
    for k in range(200):
        res = ctrl.step(state, obs)
        accel = -p.a_max if res.shield else res.accel
        state = rollout(state, np.array([[[res.steer, accel]]]), p.dt,
                        p.wheelbase, p.v_max, p.steer_max)[0, 1]
        obs[0, :2] += obs[0, 4] * p.dt
        i = tr.locate(state[0], state[1])
        _, s, d = tr.project(state[:1], state[1:2], np.array([i]))
        ahead, behind, gap_d = costs.obstacle_gaps(s, d, np.zeros(1), obs, tr,
                                                   p.half_width, p.half_length)
        hits.append(bool(np.any((np.maximum(ahead, behind) < 0) & (gap_d < 0))))
    assert not any(hits)
    assert state[3] == pytest.approx(1.0, abs=0.6)       # 앞 차 속도 근처로 추종


def test_shift_false_keeps_plan_and_shift_true_advances_it_one_step():
    p = MPPIParams(samples=1)                            # 잡음 0 후보 하나뿐 -> 계획이 그대로 남는다
    ramp = np.arange(p.horizon) * 0.1
    kept = MPPI(make_track(), p, seed=0)
    kept.u[:, 1] = ramp
    kept.step(state_on_track(1.0), NO_OBS, shift=False)
    assert np.allclose(kept.u[:, 1], ramp)
    moved = MPPI(make_track(), p, seed=0)
    moved.u[:, 1] = ramp
    moved.step(state_on_track(1.0), NO_OBS)
    assert np.allclose(moved.u[:, 1], np.append(ramp[1:], ramp[-1]))


def test_plan_respects_lateral_acceleration_limit_at_speed():
    # 기준선에서 0.8 m 벗어난 채 고속 주행: 급조향으로 돌아오면 횡가속 한계를 넘는다.
    p = MPPIParams()
    ctrl = MPPI(make_track(v_ref=6.0), p, seed=0)
    state = state_on_track(1.0, d=0.8, v=6.0)
    for _ in range(5):
        res = ctrl.step(state, NO_OBS, steer_now=0.0)
    a_lat = res.traj[:-1, 3] ** 2 * np.abs(np.tan(res.steer_plan)) / p.wheelbase
    assert a_lat.max() < p.a_lat_max * 1.15


def test_first_planned_steering_step_is_reachable_from_current_steering():
    p = MPPIParams()
    ctrl = MPPI(make_track(), p, seed=0)
    res = ctrl.step(state_on_track(1.0, d=1.0, v=2.0), NO_OBS, steer_now=-0.3)
    assert abs(res.steer_plan[0] - (-0.3)) <= p.steer_rate_max * p.dt + 1e-9


def test_terminal_weight_scales_only_last_step_tracking_cost():
    tr = make_track()
    st = state_on_track(5.0, d=0.5, v=2.0)
    traj = rollout(st, np.zeros((1, 10, 2)), 0.05, 0.33, 10.0, 0.4189)
    steer = np.zeros((1, 10))
    base = MPPI(tr, MPPIParams(horizon=10, samples=1), seed=0)
    term = MPPI(tr, MPPIParams(horizon=10, samples=1, w_terminal=4.0), seed=0)
    idx0 = tr.locate(st[0], st[1])
    c0, _ = base._stage_costs(traj, steer, idx0, NO_OBS)
    c1, _ = term._stage_costs(traj, steer, idx0, NO_OBS)
    assert np.allclose(c0[:, :-1], c1[:, :-1])
    assert c1[0, -1] > c0[0, -1]


def test_pp_steering_plan_follows_circle_and_mixing_is_optional():
    tr = make_track(v_ref=3.0)
    ctrl = MPPI(tr, MPPIParams(horizon=20, samples=64, pp_fraction=0.5), seed=0)
    st = state_on_track(5.0, d=0.0, v=3.0)
    idx0 = tr.locate(st[0], st[1])
    plan = ctrl.pp_steering_plan(st, 0.0, idx0)
    assert plan.shape == (20,)
    expected = np.arctan(0.33 / R)
    assert np.all(plan > 0.0) and abs(plan.mean() - expected) < 0.02
    res = ctrl.step(st, NO_OBS)
    assert np.isfinite(res.steer)
    # 비율 0이면 PP 열을 쓰지 않아 같은 시드에서 기존 결과와 같다.
    a = MPPI(tr, MPPIParams(horizon=20, samples=64), seed=1).step(st, NO_OBS)
    b = MPPI(tr, MPPIParams(horizon=20, samples=64, pp_fraction=0.0), seed=1).step(st, NO_OBS)
    assert a.steer == b.steer


def test_speed_profile_caps_by_upcoming_slow_segment():
    n = 400
    th = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    vx = np.full(n, 8.0)
    vx[100:140] = 2.0                       # 앞쪽 느린 구간
    tr = Track(R * np.cos(th), R * np.sin(th), vx=vx, d_left=np.full(n, 1.5), d_right=np.full(n, 1.5))
    p = MPPIParams(horizon=10, samples=8, profile_a_lat=100.0, profile_a_dec=4.0, profile_dist=12.0)
    ctrl = MPPI(tr, p, seed=0)
    far = ctrl.speed_profile(0)             # 느린 구간이 12 m 밖
    near = ctrl.speed_profile(int(95 / 100 * (tr.n * 0.25)))  # 느린 구간 바로 앞
    assert far == pytest.approx(8.0)
    assert near < 8.0
    assert ctrl.speed_profile(tr.n // 2 + 20) == pytest.approx(8.0)
    # 곡률 제한: R=10에서 a_lat 6이면 sqrt(60)=7.75 < 8
    p2 = MPPIParams(horizon=10, samples=8, profile_a_lat=6.0, profile_dist=1.0)
    assert MPPI(tr, p2, seed=0).speed_profile(0) == pytest.approx(np.sqrt(60.0), rel=0.05)


def test_offset_plans_steer_toward_their_side_and_default_is_unchanged():
    tr = make_track(v_ref=3.0, half=1.5)
    ctrl = MPPI(tr, MPPIParams(horizon=20, samples=64), seed=0)
    st = state_on_track(5.0, d=0.0, v=3.0)
    idx0 = tr.locate(st[0], st[1])
    left = ctrl.pp_steering_plan(st, 0.0, idx0, offset=0.6)    # 원 안쪽(+d, 왼쪽) 오프셋
    right = ctrl.pp_steering_plan(st, 0.0, idx0, offset=-0.6)
    assert left.mean() > right.mean()
    obs = np.array([[5.5, 6.5, -0.3, 0.3, 0.0, 0.0]])
    with_off = MPPI(tr, MPPIParams(horizon=20, samples=64, offset_fraction=0.5, offset_n=4), seed=1)
    res = with_off.step(st, obs)
    assert np.isfinite(res.steer)
    # 장애물이 없으면(offset_only_obstacles=1) 기존 결과와 같다.
    a = MPPI(tr, MPPIParams(horizon=20, samples=64), seed=2).step(st, NO_OBS)
    b = MPPI(tr, MPPIParams(horizon=20, samples=64, offset_fraction=0.5), seed=2).step(st, NO_OBS)
    assert a.steer == b.steer


def _ellipse_track(shift):
    """같은 타원 경로를 시작점만 shift개 돌려 만든 Track (s 원점이 다름)."""
    n = 360
    th = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    x = 14.0 * np.cos(th); y = 8.0 * np.sin(th)
    vx = 3.0 + 1.5 * np.sin(2 * th)
    half = 1.5 + 0.3 * np.cos(3 * th)
    r = lambda a: np.roll(a, -shift)
    return Track(r(x), r(y), vx=r(vx), d_left=r(half), d_right=r(half))


def test_seam_translation_invariance_of_step():
    """랩 경계(s=0)가 어디에 있든 같은 물리 상태에서 같은 명령이 나와야 한다(심 불연속 점검)."""
    params = MPPIParams(horizon=20, samples=128, pp_fraction=0.25, speed_profile_mode=2)
    ref = _ellipse_track(0)
    # 심 직전·직후와 트랙 중간 위치에서, 심이 다른 위치에 오도록 돌린 트랙과 비교
    for pos_idx in (3, 120, 350):
        i0 = pos_idx
        x, y = ref.x[i0], ref.y[i0]
        yaw = ref.psi[i0] + 0.05
        state = np.array([x, y, yaw, 3.0])
        res = []
        for shift in (0, 50, 200, 300):
            tr = _ellipse_track(shift)
            ctrl = MPPI(tr, params, seed=7)
            out = ctrl.step(state, NO_OBS, steer_now=0.0)
            sp = ctrl.speed_profile(ctrl._idx)
            res.append((out.steer, out.accel, sp, out.cost))
        res = np.array(res)
        # 재샘플링 위상(0.1 m 간격 시작점)이 달라 생기는 작은 차이만 허용(심 불연속이면 훨씬 큼)
        assert np.allclose(res, res[0], atol=2e-3, rtol=2e-2), res


def test_curvature_flags_default_unchanged_and_run():
    tr = make_track(v_ref=3.0)
    st = state_on_track(5.0, d=0.0, v=3.0)
    a = MPPI(tr, MPPIParams(horizon=20, samples=64), seed=3).step(st, NO_OBS)
    b = MPPI(tr, MPPIParams(horizon=20, samples=64, w_dsteer_curv=0.0, sigma_curv_gain=0.0), seed=3).step(st, NO_OBS)
    assert a.steer == b.steer and a.accel == b.accel
    c = MPPI(tr, MPPIParams(horizon=20, samples=64, w_dsteer_curv=5.0, sigma_curv_gain=2.0), seed=3).step(st, NO_OBS)
    assert np.isfinite(c.steer)


def test_obs_fallback_default_unchanged_and_switches_model_with_obstacles():
    tr = make_track(v_ref=3.0)
    st = state_on_track(5.0, d=0.0, v=3.0)
    base = MPPIParams(horizon=20, samples=64, model_st_substeps=2)
    a = MPPI(tr, base, seed=4).step(st, NO_OBS)
    b = MPPI(tr, MPPIParams(horizon=20, samples=64, model_st_substeps=2, obs_fallback=1), seed=4).step(st, NO_OBS)
    assert a.steer == b.steer and a.accel == b.accel               # 장애물이 없으면 같다
    obs = np.array([[6.5, 7.5, -0.3, 0.3, 0.0, 0.0]])
    c = MPPI(tr, MPPIParams(horizon=20, samples=64, model_st_substeps=2, obs_fallback=1), seed=4)
    r = c.step(st, obs)
    assert c._obs_mode and np.isfinite(r.steer)
