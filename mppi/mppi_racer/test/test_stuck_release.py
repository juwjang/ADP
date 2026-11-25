import unittest

import numpy as np

from mppi_racer.stuck_release import EscapeConfig, ShieldStuckRelease


class StraightTrack:
    def __init__(self):
        self.spacing = 0.1
        self.n = 1000
        self.length = 100.0
        self.d_left = np.full(self.n, 1.2)
        self.d_right = np.full(self.n, 1.2)

    def s_diff(self, a, b):
        return (a - b + self.length / 2) % self.length - self.length / 2

    def project(self, x, y, hint):
        s = np.asarray(x, dtype=float) % self.length
        d = np.asarray(y, dtype=float)
        idx = (np.round(s / self.spacing).astype(int)) % self.n
        return idx, s, d


class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.track = StraightTrack()
        self.config = EscapeConfig()
        self.release = ShieldStuckRelease(self.config, 0.155, 0.29, 0.25, 0.33, 1.5, 0.5)
        self.obs = np.array([[2.0, 2.3, -0.15, 0.15, 0.0, 0.0]])

    def step(self, i, *, s=0.0, v=0.0, shield=True, accel=8.0,
             ttc=0.25, obs=None, forced_stop=False):
        state = np.array([s, 0.0, 0.0, v])
        output = self.release.consider(i / 40.0, state, s, 0.0, 0.0, shield,
                                       accel, ttc, self.obs if obs is None else obs,
                                       self.track, forced_stop=forced_stop)
        self.release.observe(0.0 if output is None else output)
        return output

    def test_requires_stationary_hold_then_releases(self):
        for i in range(81):
            self.assertIsNone(self.step(i))
        self.assertEqual(self.step(81), 0.30)
        self.assertEqual(self.step(82), 0.30)
        self.assertIsNone(self.step(83, ttc=0.05))
        self.assertIsNone(self.step(84))

    def test_moving_low_speed_does_not_arm(self):
        for i in range(160):
            self.assertIsNone(self.step(i, s=i * 0.0015, v=0.06))

    def test_no_release_without_shield_or_positive_accel(self):
        for i in range(120):
            self.assertIsNone(self.step(i, shield=False, accel=-2.0))

    def test_close_box_and_forced_stop_block(self):
        close = np.array([[0.65, 0.95, -0.15, 0.15, 0.0, 0.0]])
        for i in range(90):
            self.assertIsNone(self.step(i, obs=close))
        for i in range(90, 180):
            self.assertIsNone(self.step(i, forced_stop=True))

    def test_predicted_gap_can_reduce_creep_speed(self):
        self.release = ShieldStuckRelease(
            EscapeConfig(pred_gap_min=0.44), 0.155, 0.29, 0.25, 0.33, 1.5, 0.5)
        near = np.array([[0.75, 1.05, -0.15, 0.15, 0.0, 0.0]])
        for i in range(81):
            self.assertIsNone(self.step(i, obs=near))
        output = self.step(81, obs=near)
        self.assertIsNotNone(output)
        self.assertGreaterEqual(output, 0.1)
        self.assertLess(output, 0.3)

    def test_release_ends_when_vehicle_reaches_creep_speed(self):
        for i in range(82):
            self.step(i)
        self.assertTrue(self.release.active)
        self.assertIsNone(self.step(82, v=0.31))
        self.assertFalse(self.release.active)


if __name__ == "__main__":
    unittest.main()
