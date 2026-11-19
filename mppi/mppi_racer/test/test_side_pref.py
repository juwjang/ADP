import numpy as np

from mppi_racer import costs
from mppi_racer.mppi import MPPI, MPPIParams
from mppi_racer.track import Track

HW = 0.155


def track():
    n = 400
    th = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    return Track(10 * np.cos(th), 10 * np.sin(th), vx=np.full(n, 3.0),
                 d_left=np.full(n, 1.0), d_right=np.full(n, 1.0))


def obs(d_lo, d_hi, s=10.0):
    return np.array([[s, s + 0.3, d_lo, d_hi, 0.0, 0.0]])


def test_side_target_is_the_wider_side_and_inside_bounds():
    tr = track()
    # 장애물이 오른쪽(-0.5..-0.2)에 있으면 왼쪽 여유가 크다 -> 목표 d > 장애물 위쪽 가장자리
    t = costs.side_targets(obs(-0.5, -0.2), tr, HW, 0.25, 0.15, 0.10)
    assert t[0] > -0.2
    assert t[0] <= 1.0 - HW - 0.10 + 1e-9
    # 왼쪽(0.2..0.5)에 있으면 오른쪽 통과
    t = costs.side_targets(obs(0.2, 0.5), tr, HW, 0.25, 0.15, 0.10)
    assert t[0] < 0.2
    assert t[0] >= -(1.0 - HW - 0.10) - 1e-9


def test_side_pref_cost_active_only_ahead_within_window_and_not_passed():
    d = np.array([[0.0, 0.0, 0.0]])
    ahead = np.array([[[3.0], [10.0], [-2.0]]])
    behind = np.array([[[-5.0], [-12.0], [1.0]]])   # 세 번째는 이미 지나침
    cost, has = costs.side_pref_cost(ahead, behind, d[..., None][..., 0], np.array([0.5]), 6.0, 10.0)
    assert has.tolist() == [[True, False, False]]
    assert cost[0, 0] == 10.0 * 0.5 ** 2
    assert cost[0, 1] == 0.0 and cost[0, 2] == 0.0


def test_side_pref_cost_uses_nearest_obstacle_target():
    d = np.array([[0.0]])
    ahead = np.array([[[4.0, 2.0]]])
    behind = np.array([[[-6.0, -4.0]]])
    cost, has = costs.side_pref_cost(ahead, behind, d, np.array([1.0, -0.5]), 6.0, 2.0)
    assert has[0, 0]
    assert np.isclose(cost[0, 0], 2.0 * 0.5 ** 2)   # 더 가까운(ahead 2) 장애물의 목표 -0.5


def _params(**kw):
    return MPPIParams(samples=256, **kw)


def test_option_off_and_no_obstacles_is_identical_to_baseline():
    tr = track()
    state = np.array([tr.x[5], tr.y[5], tr.psi[5], 3.0])
    a = MPPI(tr, _params(), seed=3).step(state, np.zeros((0, 6)))
    b = MPPI(tr, _params(side_pref_w=20.0), seed=3).step(state, np.zeros((0, 6)))
    assert np.isclose(a.steer, b.steer) and np.isclose(a.accel, b.accel) and np.isclose(a.cost, b.cost)


def test_option_off_with_obstacle_is_identical_to_baseline():
    tr = track()
    state = np.array([tr.x[5], tr.y[5], tr.psi[5], 3.0])
    o = obs(-0.6, -0.1, s=tr.s[5] + 3.0)
    a = MPPI(tr, _params(), seed=3).step(state, o)
    b = MPPI(tr, _params(side_pref_w=0.0), seed=3).step(state, o)
    assert np.isclose(a.steer, b.steer) and np.isclose(a.cost, b.cost)


def test_option_on_changes_cost_when_obstacle_is_near():
    tr = track()
    state = np.array([tr.x[5], tr.y[5], tr.psi[5], 3.0])
    o = obs(-0.6, -0.1, s=tr.s[5] + 3.0)
    a = MPPI(tr, _params(), seed=3).step(state, o)
    b = MPPI(tr, _params(side_pref_w=20.0), seed=3).step(state, o)
    assert not np.isclose(a.cost, b.cost)


def test_side_keep_off_is_identical_to_step1():
    tr = track()
    state = np.array([tr.x[5], tr.y[5], tr.psi[5], 3.0])
    o = obs(-0.6, -0.1, s=tr.s[5] + 3.0)
    a = MPPI(tr, _params(side_pref_w=30.0), seed=3).step(state, o)
    b = MPPI(tr, _params(side_pref_w=30.0, side_keep=0), seed=3).step(state, o)
    assert np.isclose(a.steer, b.steer) and np.isclose(a.cost, b.cost)


def test_side_keep_remembers_side_after_obstacle_disappears_then_expires():
    tr = track()
    state = np.array([tr.x[5], tr.y[5], tr.psi[5], 3.0])
    o = obs(-0.6, -0.1, s=tr.s[5] + 3.0)
    now = [0.0]
    m = MPPI(tr, _params(side_pref_w=30.0, side_keep=1, side_keep_ttl=1.0), seed=3)
    m._clock = lambda: now[0]
    m.step(state, o)
    assert len(m._side_mem) == 1 and m._side_sign[0] == 1.0     # 오른쪽에 있는 장애물 -> 왼쪽 통과
    now[0] = 0.5
    m.step(state, np.zeros((0, 6)))
    assert len(m._side_mem) == 1                                 # 인식에서 사라져도 기억
    now[0] = 2.0
    m.step(state, np.zeros((0, 6)))
    assert len(m._side_mem) == 0                                 # ttl 지나면 삭제


def test_side_keep_hysteresis_keeps_side_unless_other_side_much_wider():
    tr = track()
    state = np.array([tr.x[5], tr.y[5], tr.psi[5], 3.0])
    s_obs = tr.s[5] + 3.0
    m = MPPI(tr, _params(side_pref_w=30.0, side_keep=1, side_hyst=0.3), seed=3)
    m.step(state, obs(-0.5, -0.1, s=s_obs))                      # 왼쪽 여유가 큼 -> +1
    assert m._side_sign[0] == 1.0
    m.step(state, obs(-0.1, 0.3, s=s_obs))                       # 거의 비슷(차 0.4 미만) -> 유지
    assert m._side_sign[0] == 1.0
    m.step(state, obs(0.3, 0.75, s=s_obs))                       # 오른쪽이 0.3 이상 더 넓음... 매칭 거리 넘어 새 장애물
    assert m._side_sign[0] == -1.0


def test_side_pref_cost_clips_target_to_local_bounds():
    d = np.array([[0.0]])
    ahead = np.array([[[3.0]]])
    behind = np.array([[[-5.0]]])
    cost, _ = costs.side_pref_cost(ahead, behind, d, np.array([0.8]), 6.0, 10.0, lo=-0.4, hi=0.3)
    assert np.isclose(cost[0, 0], 10.0 * 0.3 ** 2)   # 목표 0.8이 0.3으로 제한됨


def test_side_pref_clip_off_is_identical_and_on_changes_cost_in_narrow_bounds():
    tr = track()
    state = np.array([tr.x[5], tr.y[5], tr.psi[5], 3.0])
    o = obs(-0.6, -0.1, s=tr.s[5] + 3.0)
    a = MPPI(tr, _params(side_pref_w=30.0), seed=3).step(state, o)
    b = MPPI(tr, _params(side_pref_w=30.0, side_pref_clip=0), seed=3).step(state, o)
    assert np.isclose(a.cost, b.cost)
    c = MPPI(tr, _params(side_pref_w=30.0, side_pref_clip=1, bounds_margin=0.6), seed=3).step(state, o)
    assert not np.isclose(a.cost, c.cost)
