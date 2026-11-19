import numpy as np

from mppi_racer.group_plan import GroupPlanner, dref_profile
from mppi_racer.mppi import MPPI, MPPIParams
from mppi_racer.track import Track

HW = 0.155


def circle(radius=10.0, half=1.0, n=400):
    th = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    return Track(radius * np.cos(th), radius * np.sin(th), vx=np.full(n, 3.0),
                 d_left=np.full(n, half), d_right=np.full(n, half))


def obs(s, d_lo, d_hi, w=0.3):
    return [s, s + w, d_lo, d_hi, 0.0, 0.0]


def test_picks_wide_side_and_clearance():
    tr = circle()
    gp = GroupPlanner(tr, HW)
    # 오른쪽에 붙은 장애물(-0.9..-0.6): 왼쪽만 넓음
    plan = gp.plan(10.0, 0.0, 3.0, np.array([obs(14.0, -0.9, -0.6)]))
    assert plan is not None and plan["sides"] == (+1,)
    assert plan["knots_d"][1] > -0.6 + HW   # 목표 d가 장애물 위쪽 가장자리 + 차폭/2 보다 큼


def test_two_obstacles_group_is_consistent_and_infeasible_side_dropped():
    tr = circle(half=1.2)
    gp = GroupPlanner(tr, HW)
    o = np.array([obs(14.0, -0.2, 0.1), obs(17.0, -0.3, 0.0)])
    plan = gp.plan(10.0, 0.0, 3.0, o)
    assert plan is not None and plan["n"] == 2
    # 두 번째 장애물이 첫 번째와 같은 쪽이면 횡이동이 적다(S자 전환 회피)
    assert plan["sides"][0] == plan["sides"][1]


def test_inner_side_rejected_in_sharp_curve():
    n = 400
    th = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    tr = Track(1.0 * np.cos(th), 1.0 * np.sin(th), vx=np.full(n, 3.0), d_left=np.full(n, 0.9), d_right=np.full(n, 0.9))
    gp = GroupPlanner(tr, HW)
    # 반지름 1 m 원(곡률 +1, 왼쪽 회전): 안쪽(왼쪽, d>0)은 곡률이 커져 탈락, 오른쪽만 가능
    plan = gp.plan(0.5, 0.0, 2.0, np.array([obs(2.0, -0.15, 0.15)]))
    assert plan is None or plan["sides"] == (-1,)


def test_no_obstacles_returns_none_and_flag_off_identical():
    tr = circle()
    gp = GroupPlanner(tr, HW)
    assert gp.plan(1.0, 0.0, 3.0, np.zeros((0, 6))) is None
    st = np.array([tr.x[0], tr.y[0], tr.psi[0], 3.0])
    a = MPPI(tr, MPPIParams(samples=64), seed=3)
    b = MPPI(tr, MPPIParams(samples=64, grp_plan=1), seed=3)
    for _ in range(5):
        ra = a.step(st, np.zeros((0, 6)))
        rb = b.step(st, np.zeros((0, 6)))
        assert ra.steer == rb.steer and ra.accel == rb.accel and ra.cost == rb.cost


def test_profile_interpolation_and_release():
    ks = np.array([0.0, 2.0, 3.0, 6.0])
    kd = np.array([0.0, 0.5, 0.5, 0.0])
    v = dref_profile(ks, kd, np.array([0.0, 1.0, 2.5, 6.0, 9.0]))
    assert v[0] == 0.0 and abs(v[1] - 0.25) < 1e-9 and v[2] == 0.5 and v[3] == 0.0 and v[4] == 0.0


def test_memory_keeps_side_until_passed():
    tr = circle(half=1.5)
    gp = GroupPlanner(tr, HW)
    o = np.array([obs(14.0, -0.1, 0.2)])
    p1 = gp.plan(10.0, 0.0, 3.0, o)
    side = p1["sides"][0]
    # 차가 반대쪽으로 약간 치우쳐도 기억된 쪽 유지
    p2 = gp.plan(11.0, -side * 0.05, 3.0, o)
    assert p2["sides"][0] == side
    # 지나간 뒤(차 s가 장애물 끝을 1 m 넘음) 기억은 정리된다
    gp.plan(16.0, 0.0, 3.0, np.zeros((0, 6)))
    assert gp.mem == []
