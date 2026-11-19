import numpy as np
import pytest

from mppi_racer import costs
from mppi_racer.track import Track

HW, HL = 0.15, 0.25


def straightish_track():
    """둘레 약 62.8 m의 큰 원(반지름 10 m)."""
    n = 400
    th = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    return Track(10 * np.cos(th), 10 * np.sin(th), vx=np.full(n, 3.0),
                 d_left=np.full(n, 1.0), d_right=np.full(n, 1.0))


def obstacle(s_start=10.0, s_end=10.3, d_right=-0.15, d_left=0.15, vs=0.0, vd=0.0):
    return np.array([[s_start, s_end, d_right, d_left, vs, vd]])


def gaps(s, d, obs, t=0.0):
    return costs.obstacle_gaps(np.atleast_1d(float(s)), np.atleast_1d(float(d)),
                               np.atleast_1d(float(t)), obs, straightish_track(), HW, HL)


# ---- 추종 / 속도 / 경계 -------------------------------------------------

def test_tracking_cost_is_zero_on_path_and_grows_quadratically():
    c = costs.tracking_cost(np.array([0.0, 0.1, 0.2]), np.zeros(3), w_d=2.0, w_heading=1.0)
    assert c[0] == 0.0
    assert c[2] == pytest.approx(4 * c[1])
    assert costs.tracking_cost(np.zeros(1), np.array([0.3]), 2.0, 1.0)[0] == pytest.approx(0.09)


def test_speed_cost_penalises_deviation_from_reference():
    c = costs.speed_cost(np.array([3.0, 2.0, 4.0]), 3.0, kappa=0.0,
                         w_speed=1.0, a_lat_max=5.0, w_alat=10.0)
    assert c[0] == 0.0
    assert c[1] == pytest.approx(1.0) and c[2] == pytest.approx(1.0)


def test_speed_cost_penalises_lateral_acceleration_over_limit_only():
    kw = dict(w_speed=0.0, a_lat_max=5.0, w_alat=10.0)
    assert costs.speed_cost(np.array([2.0]), 2.0, kappa=1.0, **kw)[0] == 0.0     # 4 m/s^2
    over = costs.speed_cost(np.array([3.0]), 3.0, kappa=-1.0, **kw)[0]           # 9 m/s^2
    assert over == pytest.approx(10.0 * 4.0 ** 2)


def test_bounds_cost_zero_inside_and_positive_near_or_past_boundary():
    kw = dict(d_left=1.0, d_right=1.0, half_width=HW, margin=0.1, w=100.0)
    c = costs.bounds_cost(np.array([0.0, 0.74, 0.85, -0.85, -1.2]), **kw)
    assert c[0] == 0.0 and c[1] == 0.0
    assert c[2] == pytest.approx(100.0 * 0.1 ** 2)
    assert c[3] == pytest.approx(c[2])
    assert c[4] > c[3]


# ---- 장애물 ------------------------------------------------------------

def test_gaps_ahead_of_static_obstacle_in_same_lane():
    ahead, behind, gap_d = gaps(8.0, 0.0, obstacle())
    assert ahead[0, 0] == pytest.approx(2.0 - HL)
    assert behind[0, 0] < 0
    assert gap_d[0, 0] == pytest.approx(-0.15 - HW)


def test_gaps_lateral_clearance_when_passing_on_the_left():
    ahead, behind, gap_d = gaps(10.1, 0.6, obstacle())
    assert max(ahead[0, 0], behind[0, 0]) < 0          # 세로로는 겹침
    assert gap_d[0, 0] == pytest.approx(0.6 - 0.15 - HW)


def test_gaps_wrap_across_start_line():
    tr = straightish_track()
    obs = obstacle(s_start=0.5, s_end=0.8)
    ahead, _, _ = gaps(tr.length - 1.0, 0.0, obs)
    assert ahead[0, 0] == pytest.approx(1.5 - HL)


def test_gaps_use_constant_velocity_prediction():
    obs = obstacle(vs=2.0, vd=0.5)
    ahead0, _, gd0 = gaps(8.0, 0.0, obs, t=0.0)
    ahead1, _, gd1 = gaps(8.0, 0.0, obs, t=1.0)
    assert ahead1[0, 0] == pytest.approx(ahead0[0, 0] + 2.0)
    assert gd1[0, 0] == pytest.approx(gd0[0, 0] + 0.5)     # 장애물이 왼쪽으로 0.5 m 이동


def test_gaps_shape_is_points_by_obstacles():
    obs = np.vstack([obstacle(), obstacle(20.0, 20.3)])
    s = np.full((5, 7), 8.0)
    ahead, behind, gap_d = costs.obstacle_gaps(s, np.zeros((5, 7)), np.zeros((5, 7)),
                                               obs, straightish_track(), HW, HL)
    assert ahead.shape == behind.shape == gap_d.shape == (5, 7, 2)


def test_obstacle_cost_collision_far_and_near():
    kw = dict(margin_s=0.3, margin_d=0.3, w_collision=1000.0, w_near=10.0)

    def cost(s, d):
        return costs.obstacle_cost(*gaps(s, d, obstacle()), **kw)[0]

    assert cost(10.1, 0.0) >= 1000.0                    # 충돌
    assert cost(5.0, 0.0) == 0.0                        # 한참 뒤
    assert cost(10.1, 0.9) == 0.0                       # 옆으로 충분히 떨어짐
    near = cost(10.1, 0.45)                             # 옆 여유 0.15 m
    assert 0.0 < near < 1000.0
    assert cost(10.1, 0.35) > near                      # 가까울수록 큼


def test_obstacle_cost_sums_over_obstacles_and_handles_none():
    kw = dict(margin_s=0.3, margin_d=0.3, w_collision=1000.0, w_near=10.0)
    empty = np.zeros((0, 6))
    g = costs.obstacle_gaps(np.array([1.0]), np.array([0.0]), np.array([0.0]),
                            empty, straightish_track(), HW, HL)
    assert costs.obstacle_cost(*g, **kw)[0] == 0.0
    assert np.isinf(costs.time_to_collision(g[0], g[2], np.array([3.0]), empty[:, 4])[0])


# ---- TTC ---------------------------------------------------------------

def test_ttc_is_gap_over_closing_speed():
    obs = obstacle(vs=1.0)
    ahead, _, gap_d = gaps(8.0, 0.0, obs)
    ttc = costs.time_to_collision(ahead, gap_d, np.array([3.0]), obs[:, 4])
    assert ttc[0] == pytest.approx((2.0 - HL) / 2.0)


def test_ttc_infinite_when_not_closing_or_laterally_clear_or_behind():
    obs = obstacle(vs=4.0)
    ahead, _, gap_d = gaps(8.0, 0.0, obs)
    assert np.isinf(costs.time_to_collision(ahead, gap_d, np.array([3.0]), obs[:, 4])[0])
    obs = obstacle()
    ahead, _, gap_d = gaps(8.0, 0.7, obs)
    assert np.isinf(costs.time_to_collision(ahead, gap_d, np.array([3.0]), obs[:, 4])[0])
    ahead, _, gap_d = gaps(12.0, 0.0, obs)
    assert np.isinf(costs.time_to_collision(ahead, gap_d, np.array([3.0]), obs[:, 4])[0])


def test_ttc_takes_minimum_over_obstacles():
    obs = np.vstack([obstacle(12.0, 12.3), obstacle(10.0, 10.3)])
    ahead, _, gap_d = gaps(8.0, 0.0, obs)
    ttc = costs.time_to_collision(ahead, gap_d, np.array([2.0]), obs[:, 4])
    assert ttc[0] == pytest.approx((2.0 - HL) / 2.0)


def test_ttc_cost_zero_above_threshold_and_increasing_below():
    c = costs.ttc_cost(np.array([np.inf, 1.5, 1.0, 0.5, 0.0]), threshold=1.5, w=9.0)
    assert c[0] == 0.0 and c[1] == 0.0
    assert 0.0 < c[2] < c[3] < c[4]
    assert c[4] == pytest.approx(9.0)


# ---- 입력 변화량 / shield ------------------------------------------------

def test_smoothness_cost_zero_for_constant_input_matching_previous():
    u = np.tile(np.array([0.1, 1.0]), (3, 10, 1))
    c = costs.smoothness_cost(u, np.array([0.1, 1.0]), w_dsteer=5.0, w_daccel=1.0)
    assert np.allclose(c, 0.0)


def test_smoothness_cost_counts_jump_from_previous_and_within_horizon():
    u = np.zeros((1, 4, 2))
    u[0, 2:, 0] = 0.2                                   # 호라이즌 안에서 조향 1회 변화
    c = costs.smoothness_cost(u, np.array([0.0, 2.0]), w_dsteer=5.0, w_daccel=1.0)
    assert c[0] == pytest.approx(5.0 * 0.2 ** 2 + 1.0 * 2.0 ** 2)


def test_shield_limits_speed_only_below_ttc_floor():
    assert costs.shield_speed(3.0, min_ttc=0.8, ttc_floor=0.5, safe_speed=0.0) == 3.0
    assert costs.shield_speed(3.0, min_ttc=0.3, ttc_floor=0.5, safe_speed=0.0) == 0.0
    assert costs.shield_speed(0.5, min_ttc=0.3, ttc_floor=0.5, safe_speed=1.0) == 0.5


# ---- 차량 자체의 횡가속도 --------------------------------------------------

def test_lateral_accel_cost_zero_below_limit_and_quadratic_above():
    L = 0.33
    kw = dict(wheelbase=L, a_lat_max=6.0, w=2.0)
    steer = np.arctan(L * 0.5)                           # 곡률 0.5 1/m
    assert costs.lateral_accel_cost(np.array([3.0]), np.array([steer]), **kw)[0] == 0.0   # 4.5
    over = costs.lateral_accel_cost(np.array([4.0]), np.array([-steer]), **kw)[0]         # 8.0
    assert over == pytest.approx(2.0 * 2.0 ** 2)


def test_speed_cost_asymmetric_and_gain_defaults_unchanged():
    import numpy as np
    from mppi_racer import costs
    v = np.array([3.0, 5.0, 7.0])
    base = costs.speed_cost(v, 5.0, 0.0, 2.0, 8.0, 5.0)
    same = costs.speed_cost(v, 5.0, 0.0, 2.0, 8.0, 5.0, w_under=None, vref_gain=1.0)
    assert np.allclose(base, same)
    asym = costs.speed_cost(v, 5.0, 0.0, 2.0, 8.0, 5.0, w_under=10.0)
    assert asym[0] == 10.0 * 4.0 and asym[2] == base[2]     # 미달은 10배/2배 가중, 초과는 그대로
    gain = costs.speed_cost(v, 5.0, 0.0, 2.0, 8.0, 5.0, vref_gain=1.2)
    assert gain[1] == 2.0 * (5.0 - 6.0) ** 2 * 5.0 / 5.0 * 1.0 or gain[1] > base[1]
