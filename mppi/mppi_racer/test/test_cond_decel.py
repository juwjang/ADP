import numpy as np
from mppi_racer.cond_decel import cond_decel_cap
from mppi_racer.track import Track

HW, HL = 0.155, 0.29


def track():
    n = 400
    th = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    return Track(10 * np.cos(th), 10 * np.sin(th), vx=np.full(n, 3.0), d_left=np.full(n, 1.5), d_right=np.full(n, 1.5))


def obs(s, d_lo=-0.15, d_hi=0.15):
    return np.array([[s, s + 0.3, d_lo, d_hi, 0.0, 0.0]])


def test_no_obstacle_no_cap():
    assert cond_decel_cap(track(), 5.0, 0.0, np.zeros((0, 6)), HW, HL) == np.inf


def test_cap_decreases_with_distance_and_matches_formula():
    tr = track()
    far = cond_decel_cap(tr, 5.0, 0.0, obs(11.0), HW, HL)
    near = cond_decel_cap(tr, 5.0, 0.0, obs(8.0), HW, HL)
    assert near < far
    ahead = 11.0 - 5.0 - HL
    assert np.isclose(far, np.sqrt(2 * 2.5 * (ahead - 0.8)) + 1.5)


def test_released_after_passing_or_alongside():
    tr = track()
    assert cond_decel_cap(tr, 5.0, 0.0, obs(5.2), HW, HL) == np.inf      # 앞끝이 장애물 뒤끝에 닿음/나란히
    assert cond_decel_cap(tr, 9.0, 0.0, obs(5.0), HW, HL) == np.inf      # 이미 지나침


def test_laterally_clear_obstacle_ignored():
    tr = track()
    assert cond_decel_cap(tr, 5.0, -1.0, obs(9.0, 0.3, 0.6), HW, HL) == np.inf


def test_combined_with_shield_takes_lower_value():
    tr = track()
    cap = cond_decel_cap(tr, 5.0, 0.0, obs(9.0), HW, HL)
    for shield in (0.5, 3.0, 10.0):
        assert min(shield, cap) == (shield if shield < cap else cap)
    assert min(1.5, cap) <= 1.5 and min(1.5, cap) <= cap
