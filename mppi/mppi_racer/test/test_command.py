import pytest

from mppi_racer.command import speed_command

KW = dict(k_accel=4.0, k_brake=20.0, v_max=10.0)


def test_zero_acceleration_commands_current_speed():
    assert speed_command(3.0, 0.0, **KW) == pytest.approx(3.0)


def test_acceleration_uses_accel_gain():
    assert speed_command(3.0, 2.0, **KW) == pytest.approx(3.5)


def test_braking_uses_brake_gain():
    assert speed_command(3.0, -4.0, **KW) == pytest.approx(2.8)


def test_command_is_clipped_to_zero_and_v_max():
    assert speed_command(0.1, -8.0, **KW) == 0.0
    assert speed_command(9.9, 8.0, **KW) == 10.0


def test_lowpass_alpha_one_is_identity_and_alpha_half_averages():
    from mppi_racer.command import lowpass
    assert lowpass(0.3, -0.1, 1.0) == pytest.approx(-0.1)
    assert lowpass(0.3, -0.1, 0.5) == pytest.approx(0.1)
    assert lowpass(0.3, -0.1, 0.0) == pytest.approx(0.3)


def test_predict_state_zero_delay_returns_input_and_straight_motion_advances():
    import numpy as np
    from mppi_racer.command import predict_state
    kw = dict(wheelbase=0.33, v_max=10.0, steer_max=0.4189, steer_rate_max=3.2)
    s0 = np.array([1.0, 2.0, 0.0, 5.0])
    s, st = predict_state(s0, 0.05, 0.1, 0.0, 0.0, **kw)
    assert np.allclose(s, s0) and st == 0.05
    s, st = predict_state(s0, 0.0, 0.0, 2.0, 0.1, **kw)
    assert s[0] == pytest.approx(1.5) and s[1] == pytest.approx(2.0)
    assert s[3] == pytest.approx(5.2) and st == 0.0
