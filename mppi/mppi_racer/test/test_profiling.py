import numpy as np

from mppi_racer.mppi import MPPI, MPPIParams
from mppi_racer.track import Track


def test_optional_profile_preserves_commands_and_accounts_for_step():
    theta = np.linspace(0, 2 * np.pi, 100, endpoint=False)
    tr = Track(10*np.cos(theta), 10*np.sin(theta), np.full(100, 3.),
               np.full(100, 1.5), np.full(100, 1.5))
    p = MPPIParams(samples=32, horizon=10)
    plain, measured = MPPI(tr, p, seed=4), MPPI(tr, p, seed=4)
    state = [10., 0., np.pi/2, 2.]
    for _ in range(3):
        timings = {'stale': 123}
        a = plain.step(state, [])
        b = measured.step(state, [], profile=timings)
        assert 'stale' not in timings
        assert set(timings) == {'prepare_ms', 'sample_ms', 'rollout_ms',
                                'cost_ms', 'update_ms', 'final_ms', 'total_ms'}
        assert all(np.isfinite(v) and v >= 0 for v in timings.values())
        assert sum(v for k, v in timings.items() if k != 'total_ms') <= timings['total_ms'] + 1e-6
        assert a.steer == b.steer and a.accel == b.accel
        np.testing.assert_array_equal(a.traj, b.traj)
