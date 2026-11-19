import pytest
from mppi_racer import command


def test_new_command_does_not_act_during_previous_interval():
    observer = command.SteeringObserver(rate=2.0)
    observer.command(.4, 10.0)
    assert observer.estimate(10.1) == pytest.approx(.2)
    # During planning the old +.4 command remains active.
    observer.command(-.4, 10.15)
    assert observer.angle == pytest.approx(.3)
    assert observer.estimate(10.2) == pytest.approx(.2)


def test_observer_reaches_target_without_overshoot_after_long_interval():
    observer = command.SteeringObserver(rate=3.2)
    observer.command(.1, 1.)
    assert observer.estimate(2.) == pytest.approx(.1)
    assert observer.estimate(2.1) == pytest.approx(.1)
