import numpy as np
from mppi_racer.raw_confirm import merge_obstacles
from mppi_racer.track import Track


def track():
    n = 400
    th = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    return Track(10 * np.cos(th), 10 * np.sin(th), vx=np.full(n, 3.0), d_left=np.full(n, 1.5), d_right=np.full(n, 1.5))


def test_merge_prefers_published_and_adds_new_raw():
    tr = track()
    pub = np.array([[10.0, 10.25, 0.1, 0.35, 0.0, 0.0]])
    conf = np.array([[10.05, 10.3, 0.12, 0.36, 0.0, 0.0], [20.0, 20.25, -0.5, -0.25, 0.0, 0.0]])
    out = merge_obstacles(pub, conf, tr)
    assert out.shape == (2, 6) and np.allclose(out[0], pub[0]) and np.isclose(out[1, 0], 20.0)


def test_merge_empty_cases():
    tr = track()
    assert merge_obstacles(np.zeros((0, 6)), np.zeros((0, 6)), tr).shape == (0, 6)
    conf = np.array([[10.0, 10.25, 0.1, 0.35, 0.0, 0.0]])
    assert merge_obstacles(np.zeros((0, 6)), conf, tr).shape == (1, 6)
