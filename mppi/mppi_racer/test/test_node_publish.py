"""Run with the ROS workspace sourced; pure numerical suites may skip this file."""
from types import SimpleNamespace
import pytest

pytest.importorskip('rclpy')
from builtin_interfaces.msg import Time
from mppi_racer import node
from mppi_racer.command import SteeringObserver


def test_emergency_stop_updates_observer_command(monkeypatch):
    observer = SteeringObserver(2.)
    observer.command(.4, 0.)
    published = []
    fake = SimpleNamespace(
        _steering=observer,
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(to_msg=lambda: Time())),
        drive_pub=SimpleNamespace(publish=published.append))
    monkeypatch.setattr(node.time, 'monotonic', lambda: .1)
    node.MPPIRacer.publish(fake, 0., 0.)
    assert published[0].drive.speed == 0.
    assert observer.target == 0.
    assert observer.estimate(.15) == pytest.approx(.1)
