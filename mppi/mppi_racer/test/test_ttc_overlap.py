import numpy as np
from mppi_racer.costs import time_to_collision


def test_existing_overlap_is_zero_ttc_even_when_stopped():
    ttc = time_to_collision(np.array([[-.1]]), np.array([[-.05]]),
                            np.array([0.]), np.array([0.]), behind=np.array([[-.2]]))
    assert ttc[0] == 0.0


def test_passed_obstacle_is_not_existing_collision():
    ttc = time_to_collision(np.array([[-2.]]), np.array([[-.05]]),
                            np.array([3.]), np.array([0.]), behind=np.array([[1.5]]))
    assert np.isinf(ttc[0])
