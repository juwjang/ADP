import numpy as np
import pytest

from mppi_racer.model import rollout

L = 0.33
LIM = dict(wheelbase=L, v_max=10.0, steer_max=0.4189)


def test_rollout_shape_and_initial_state():
    state = np.array([1.0, 2.0, 0.5, 3.0])
    traj = rollout(state, np.zeros((7, 30, 2)), 0.05, **LIM)
    assert traj.shape == (7, 31, 4)
    assert np.allclose(traj[:, 0], state)


def test_zero_input_goes_straight_at_constant_speed():
    state = np.array([0.0, 0.0, np.pi / 2, 2.0])
    traj = rollout(state, np.zeros((1, 20, 2)), 0.05, **LIM)
    assert traj[0, -1, 0] == pytest.approx(0.0, abs=1e-9)
    assert traj[0, -1, 1] == pytest.approx(2.0, abs=1e-9)
    assert traj[0, -1, 3] == pytest.approx(2.0)


def test_constant_steer_follows_circle_of_bicycle_radius():
    delta, v = 0.2, 2.0
    radius = L / np.tan(delta)
    u = np.zeros((1, 400, 2))
    u[..., 0] = delta
    traj = rollout(np.array([0.0, 0.0, 0.0, v]), u, 0.005, **LIM)
    # 좌회전 원의 중심은 (0, radius)
    r = np.hypot(traj[0, :, 0], traj[0, :, 1] - radius)
    assert np.allclose(r, radius, atol=0.01)
    assert traj[0, -1, 2] == pytest.approx(v / radius * 2.0, rel=1e-3)


def test_acceleration_integrates_speed():
    u = np.zeros((1, 10, 2))
    u[..., 1] = 2.0
    traj = rollout(np.array([0.0, 0.0, 0.0, 1.0]), u, 0.1, **LIM)
    assert traj[0, -1, 3] == pytest.approx(3.0)


def test_speed_is_clipped_between_zero_and_v_max():
    u = np.zeros((2, 50, 2))
    u[0, :, 1] = 9.0
    u[1, :, 1] = -9.0
    traj = rollout(np.array([0.0, 0.0, 0.0, 5.0]), u, 0.1, **LIM)
    assert traj[0, :, 3].max() == pytest.approx(10.0)
    assert traj[1, :, 3].min() == pytest.approx(0.0)


def test_steering_is_clipped_to_limit():
    u = np.zeros((2, 20, 2))
    u[0, :, 0] = 5.0
    u[1, :, 0] = LIM['steer_max']
    traj = rollout(np.array([0.0, 0.0, 0.0, 2.0]), u, 0.05, **LIM)
    assert np.allclose(traj[0], traj[1])


# ---- 조향 속도 제한 ------------------------------------------------------

from mppi_racer.model import rollout_with_steering_lag  # noqa: E402


def test_steering_lag_limits_rate_of_actual_steering():
    u = np.zeros((1, 5, 2))
    u[..., 0] = 0.4
    _, steer = rollout_with_steering_lag(np.array([0.0, 0.0, 0.0, 2.0]), 0.0, u, 0.05,
                                         steer_rate_max=3.2, **LIM)
    assert steer.shape == (1, 5)
    assert np.allclose(steer[0], [0.16, 0.32, 0.4, 0.4, 0.4])


def test_steering_lag_starts_from_current_steering_angle():
    u = np.zeros((1, 3, 2))
    _, steer = rollout_with_steering_lag(np.array([0.0, 0.0, 0.0, 2.0]), 0.3, u, 0.05,
                                         steer_rate_max=3.2, **LIM)
    assert np.allclose(steer[0], [0.14, 0.0, 0.0])


def test_steering_lag_turns_less_than_instant_steering():
    u = np.zeros((1, 6, 2))
    u[..., 0] = 0.4
    state = np.array([0.0, 0.0, 0.0, 3.0])
    lagged, _ = rollout_with_steering_lag(state, 0.0, u, 0.05, steer_rate_max=3.2, **LIM)
    instant = rollout(state, u, 0.05, **LIM)
    assert 0.0 < lagged[0, -1, 2] < instant[0, -1, 2]


def test_infinite_steering_rate_matches_plain_rollout():
    rng = np.random.default_rng(0)
    u = rng.normal(size=(4, 10, 2)) * (0.2, 2.0)
    state = np.array([1.0, 2.0, 0.3, 3.0])
    lagged, _ = rollout_with_steering_lag(state, 0.0, u, 0.05, steer_rate_max=np.inf, **LIM)
    assert np.allclose(lagged, rollout(state, u, 0.05, **LIM))


def test_understeer_reduces_yaw_rate_only_when_enabled():
    import numpy as np
    from mppi_racer.model import rollout_with_steering_lag
    state = np.array([0.0, 0.0, 0.0, 6.0])
    u = np.zeros((1, 10, 2))
    u[..., 0] = 0.1
    kw = dict(dt=0.05, wheelbase=0.33, v_max=10.0, steer_max=0.4189, steer_rate_max=np.inf)
    base, _ = rollout_with_steering_lag(state, 0.1, u, **kw)
    same, _ = rollout_with_steering_lag(state, 0.1, u, understeer=0.0, **kw)
    under, _ = rollout_with_steering_lag(state, 0.1, u, understeer=0.04, **kw)
    assert np.allclose(base, same)
    assert 0 < under[0, -1, 2] < base[0, -1, 2]
    assert under[0, -1, 2] / base[0, -1, 2] == pytest.approx(1 - 0.04 * 6.0, rel=1e-6)


def test_rollout_st_straight_line_and_matches_kinematics_at_low_speed():
    import numpy as np
    from mppi_racer.model import rollout_st, rollout_with_steering_lag
    st = np.array([0.0, 0.0, 0.0, 5.0])
    ctrl = np.zeros((4, 20, 2))
    traj, steer = rollout_st(st, 0.0, ctrl, 0.05, 10.0, 0.4189, 3.2, substeps=2)
    assert traj.shape == (4, 21, 4) and steer.shape == (4, 20)
    assert np.allclose(traj[:, :, 1], 0.0, atol=1e-9) and np.allclose(traj[:, -1, 3], 5.0)
    assert traj[0, -1, 0] == pytest.approx(5.0, rel=1e-6)
    # 같은 작은 조향 입력에서 ST와 운동학 요(yaw) 변화는 같은 부호이고 크기가 비슷하다(ST가 약간 작음)
    ctrl[..., 0] = 0.05
    a, _ = rollout_st(st, 0.05, ctrl, 0.05, 10.0, 0.4189, 3.2, substeps=2)
    b, _ = rollout_with_steering_lag(st, 0.05, ctrl, 0.05, 0.33, 10.0, 0.4189, 3.2)
    assert a[0, -1, 2] > 0 and 0.4 * b[0, -1, 2] < a[0, -1, 2] < 1.3 * b[0, -1, 2]


def test_rollout_st_fast_matches_numpy_version():
    import numpy as np
    from mppi_racer import model
    if model.rollout_st_fast is None:
        pytest.skip('numba 없음')
    rng = np.random.default_rng(1)
    c = np.zeros((16, 20, 2)); c[..., 0] = rng.normal(0, 0.1, (16, 20)); c[..., 1] = rng.normal(0, 2.5, (16, 20))
    st = np.array([1.0, 2.0, 0.3, 4.0])
    a, sa = model.rollout_st(st, 0.02, c, 0.05, 10.0, 0.4189, 3.2, substeps=2)
    b, sb = model.rollout_st_fast(st, 0.02, c, 0.05, 10.0, 0.4189, 3.2, substeps=2)
    assert np.allclose(a, b, atol=1e-6) and np.allclose(sa, sb, atol=1e-8)
