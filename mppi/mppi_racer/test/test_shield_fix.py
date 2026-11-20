from mppi_racer.costs import shield_command


def test_fix_off_is_old_behavior():
    for v in (0.0, 0.5, 1.5, 3.0, 6.0):
        for vp in (0.0, 0.4, 2.4, 9.0):
            assert shield_command(v, vp, 1.5, fix=0) == min(v, 1.5)


def test_fix_allows_start_from_standstill():
    assert shield_command(0.0, 2.4, 1.5, fix=1) == 1.5
    assert abs(shield_command(0.0, 0.3, 1.5, fix=1) - 0.3) < 1e-12
    assert shield_command(0.0, 0.0, 1.5, fix=1) == 0.0


def test_fix_identical_when_moving_fast_or_plan_slower():
    # v가 safe 이상이면 기존과 같다(고속 쉴드 감속 유지)
    for v in (1.5, 2.0, 5.0):
        for vp in (0.0, 1.0, 9.0):
            assert shield_command(v, vp, 1.5, fix=1) == min(v, 1.5)
    # 저속에서 계획 속도가 현재보다 느리면 현재 속도 유지(기존 min(v, safe))
    assert shield_command(1.0, 0.2, 1.5, fix=1) == 1.0
