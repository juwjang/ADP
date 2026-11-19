from f110_msgs.msg import Wpnt, WpntArray
from speed_scaler.speed_scaler import SpeedScaler


def test_repeated_scaling_keeps_original_and_does_not_compound():
    source = WpntArray()
    waypoint = Wpnt()
    waypoint.vx_mps = 10.0
    waypoint.x_m = 2.0
    source.wpnts = [waypoint]
    first = SpeedScaler.scale(None, source, .7)
    second = SpeedScaler.scale(None, source, .7)
    assert source.wpnts[0].vx_mps == 10.0
    assert first.wpnts[0].vx_mps == second.wpnts[0].vx_mps == 7.0
    first.wpnts[0].x_m = 99.0
    assert source.wpnts[0].x_m == second.wpnts[0].x_m == 2.0


def test_stop_then_resume_recovers_reference_speed():
    source = WpntArray()
    waypoint = Wpnt()
    waypoint.vx_mps = 10.0
    source.wpnts = [waypoint]
    stopped = SpeedScaler.scale(None, source, 0.0)
    resumed = SpeedScaler.scale(None, source, .7)
    assert stopped.wpnts[0].vx_mps == 0.0
    assert resumed.wpnts[0].vx_mps == 7.0
