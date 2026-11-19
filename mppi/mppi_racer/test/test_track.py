import numpy as np
import pytest

from mppi_racer.track import Track

R = 5.0


def circle_track(n=80, spacing=0.1):
    """반시계 방향 원형 트랙 (반지름 5 m). 왼쪽(+d)이 원 안쪽이다."""
    th = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    return Track(R * np.cos(th), R * np.sin(th),
                 vx=np.full(n, 3.0), d_left=np.full(n, 1.2),
                 d_right=np.full(n, 0.8), spacing=spacing)


def test_resampled_track_length_and_spacing():
    tr = circle_track()
    assert tr.length == pytest.approx(2 * np.pi * R, rel=2e-3)
    assert tr.spacing == pytest.approx(0.1, rel=0.01)
    assert tr.n == round(tr.length / tr.spacing)


def test_heading_and_curvature_of_circle():
    tr = circle_track()
    i = tr.locate(0.0, R)                      # 원의 꼭대기: 진행 방향은 -x
    assert abs(abs(tr.psi[i]) - np.pi) < 0.03
    assert np.allclose(tr.kappa, 1.0 / R, atol=0.01)


def test_locate_returns_nearest_index():
    tr = circle_track()
    i = tr.locate(R + 0.3, 0.05)
    assert np.hypot(tr.x[i] - R, tr.y[i]) < 0.12


def test_project_point_inside_circle_has_positive_d():
    tr = circle_track()
    th = 1.0
    x = np.array([(R - 0.5) * np.cos(th), (R + 0.3) * np.cos(th)])
    y = np.array([(R - 0.5) * np.sin(th), (R + 0.3) * np.sin(th)])
    hint = np.full(2, tr.locate(R * np.cos(th), R * np.sin(th)))
    idx, s, d = tr.project(x, y, hint)
    assert d[0] == pytest.approx(0.5, abs=0.01)
    assert d[1] == pytest.approx(-0.3, abs=0.01)
    assert s[0] == pytest.approx(R * th, abs=0.03)


def test_project_follows_hint_forward_across_lap_boundary():
    tr = circle_track()
    th = -0.02                                  # 출발선 직전
    hint = np.array([tr.n - 4])
    x = np.array([R * np.cos(0.05)])            # 출발선을 넘어간 점
    y = np.array([R * np.sin(0.05)])
    idx, s, d = tr.project(x, y, hint)
    assert idx[0] < 10
    assert s[0] == pytest.approx(R * 0.05, abs=0.03)
    assert abs(d[0]) < 0.01


def test_project_keeps_input_shape():
    tr = circle_track()
    x = np.full((4, 3), R)
    y = np.zeros((4, 3))
    idx, s, d = tr.project(x, y, np.zeros((4, 3), dtype=int))
    assert idx.shape == s.shape == d.shape == (4, 3)


def test_speed_and_bounds_are_interpolated():
    n = 80
    th = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    vx = np.where(th < np.pi, 2.0, 4.0)
    tr = Track(R * np.cos(th), R * np.sin(th), vx=vx,
               d_left=np.full(n, 1.2), d_right=np.full(n, 0.8))
    assert tr.vx[tr.locate(0.0, R)] == pytest.approx(2.0)
    assert tr.vx[tr.locate(0.0, -R)] == pytest.approx(4.0)
    assert np.allclose(tr.d_left, 1.2) and np.allclose(tr.d_right, 0.8)


def test_s_diff_wraps_to_shortest_signed_distance():
    tr = circle_track()
    assert tr.s_diff(0.5, tr.length - 0.5) == pytest.approx(1.0)
    assert tr.s_diff(tr.length - 0.5, 0.5) == pytest.approx(-1.0)


def test_sight_distance_straight_is_max_and_circle_is_limited():
    import numpy as np
    from mppi_racer.track import Track, sight_distance
    n = 400
    th = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    circle = Track(10 * np.cos(th), 10 * np.sin(th), vx=np.full(n, 3.0),
                   d_left=np.full(n, 1.5), d_right=np.full(n, 1.5))
    s = sight_distance(circle, max_dist=15.0)
    assert 7.0 <= s.min() <= 13.0
    assert s.max() - s.min() < 1.5          # 원은 모든 점에서 같다
    wide = Track(10 * np.cos(th), 10 * np.sin(th), vx=np.full(n, 3.0),
                 d_left=np.full(n, 9.0), d_right=np.full(n, 9.0))
    assert sight_distance(wide, max_dist=15.0).min() == pytest.approx(15.0, abs=0.6)
