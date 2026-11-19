import numpy as np
from mppi_racer.raw_confirm import RawConfirmer
from mppi_racer.track import Track


def track():
    n = 400
    th = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    return Track(10 * np.cos(th), 10 * np.sin(th), vx=np.full(n, 3.0), d_left=np.full(n, 1.5), d_right=np.full(n, 1.5))


DET = np.array([[10.0, 10.25, 0.1, 0.35]])


def test_confirms_after_n_consecutive_frames():
    rc = RawConfirmer(track(), confirm_frames=3)
    for k in range(2):
        rc.update(0.025 * k, DET)
        assert len(rc.confirmed()) == 0
    rc.update(0.05, DET)
    out = rc.confirmed()
    assert out.shape == (1, 6) and np.isclose(out[0, 0], 10.0, atol=0.05) and out[0, 4] == 0.0


def test_rejects_wrong_size_and_outside_bounds():
    rc = RawConfirmer(track(), confirm_frames=2)
    small = np.array([[10.0, 10.01, 0.1, 0.11]])
    wall = np.array([[10.0, 10.25, 1.45, 1.7]])
    for k in range(5):
        rc.update(0.025 * k, small)
        rc.update(0.025 * k + 0.001, wall)
    assert len(rc.confirmed()) == 0


def test_gap_resets_count_and_ttl_drops_confirmed():
    rc = RawConfirmer(track(), confirm_frames=3, ttl_s=1.0)
    rc.update(0.0, DET); rc.update(0.025, DET)
    rc.update(0.5, DET)                      # 끊겼다가 다시: 카운트 재시작
    assert len(rc.confirmed()) == 0
    for k in range(3):
        rc.update(0.6 + 0.025 * k, DET)
    assert len(rc.confirmed()) == 1
    rc.update(2.0, np.zeros((0, 4)))         # 1 s 넘게 못 봄 -> 삭제
    assert len(rc.confirmed()) == 0


def test_removed_after_car_passes():
    tr = track()
    rc = RawConfirmer(tr, confirm_frames=2)
    for k in range(3):
        rc.update(0.025 * k, DET, car_s=5.0)
    assert len(rc.confirmed()) == 1
    rc.update(0.1, DET, car_s=13.0)
    assert len(rc.confirmed()) == 0


def test_two_obstacles_tracked_separately():
    rc = RawConfirmer(track(), confirm_frames=2)
    two = np.array([[10.0, 10.25, 0.1, 0.35], [12.0, 12.25, -0.6, -0.35]])
    for k in range(3):
        rc.update(0.025 * k, two)
    assert len(rc.confirmed()) == 2
