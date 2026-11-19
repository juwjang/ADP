import numpy as np

from mppi_racer import costs
from mppi_racer.mppi import MPPIParams

HW, MG, W = 0.155, 0.25, 800.0


def test_zero_inside_margin_and_zero_when_just_touching_margin():
    d = np.array([[0.0, 0.3, 0.6]])  # d_left=1.0 -> 차체 한계 0.845, 여유 소진선 0.845-0.25 에서는 벽 접촉 아님
    c = costs.wall_hit_cost(d, np.full(3, 1.0), np.full(3, 1.0), HW, MG, W)
    assert np.all(c == 0.0)


def test_penalty_when_body_crosses_boundary_both_sides():
    dl = np.array([[0.8, 1.2]])
    c = costs.wall_hit_cost(dl, np.full(2, 1.0), np.full(2, 1.0), HW, MG, W)
    assert c[0, 0] == 0.0 and c[0, 1] == W
    c = costs.wall_hit_cost(-dl, np.full(2, 1.0), np.full(2, 1.0), HW, MG, W)
    assert c[0, 0] == 0.0 and c[0, 1] == W


def test_default_flag_is_off():
    assert MPPIParams().wall_hit_w == 0.0
