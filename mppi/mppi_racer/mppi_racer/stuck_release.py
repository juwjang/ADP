"""Opt-in, bounded shield-stall release using observed obstacle boxes."""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class EscapeConfig:
    hold_s: float = 2.0
    stationary_s_max: float = 0.05
    v_max: float = 0.10
    accel_min: float = 1.0
    ttc_min: float = 0.08
    ahead_min: float = 0.40
    corridor_min: float = 0.40
    wall_min: float = 0.03
    pred_gap_min: float = 0.35
    speed: float = 0.30
    speed_min: float = 0.10
    predict_s: float = 0.10


class ShieldStuckRelease:
    """A timer and a one-step geometry check; never changes the MPPI plan."""

    def __init__(self, config, half_width, half_length, bounds_margin, wheelbase,
                 shield_speed, ttc_floor):
        self.config = config
        self.half_width = float(half_width)
        self.half_length = float(half_length)
        self.bounds_margin = float(bounds_margin)
        self.wheelbase = float(wheelbase)
        self.shield_speed = float(shield_speed)
        self.ttc_floor = float(ttc_floor)
        if not (0 < config.ttc_min < self.ttc_floor and
                0 < config.speed_min <= config.speed <= self.shield_speed and
                config.hold_s > 0 and config.predict_s > 0 and
                config.ahead_min > 0 and config.pred_gap_min > 0):
            raise ValueError("invalid shield-stuck release parameters")
        self.hold_start = None
        self.hold_pos_s = None
        self.last_t = None
        self.last_speed = None
        self.active = False
        self.last_details = {}

    def reset(self):
        self.hold_start = None
        self.hold_pos_s = None
        self.active = False

    def observe(self, speed):
        self.last_speed = float(speed)

    def consider(self, now, state, car_s, car_d, steer, shield, accel, min_ttc,
                 obstacles, track, forced_stop=False):
        """Return creep speed or None. All distances are nominal Frenet box gaps."""
        c = self.config
        now = float(now)
        v = max(float(state[3]), 0.0)
        if self.last_t is not None and (now <= self.last_t or now - self.last_t > 0.15):
            self.reset()
        self.last_t = now
        if (not shield or forced_stop or len(obstacles) == 0 or
                not np.isfinite(accel) or accel < c.accel_min):
            self.reset()
            return None
        if not self.active:
            if (v > c.v_max or self.last_speed is None or
                    self.last_speed > c.v_max or min_ttc >= self.ttc_floor):
                self.reset()
                return None
            if self.hold_start is None:
                self.hold_start = now
                self.hold_pos_s = float(car_s)
            elif abs(float(track.s_diff(car_s, self.hold_pos_s))) > c.stationary_s_max:
                self.hold_start = now
                self.hold_pos_s = float(car_s)
            if now - self.hold_start < c.hold_s:
                return None
        elif v >= c.speed:
            self.reset()
            return None
        if not np.isfinite(min_ttc) or min_ttc < c.ttc_min:
            if self.active:
                self.reset()
            return None
        for creep_speed in np.linspace(c.speed, c.speed_min, 5):
            ok, details = self._safe(state, car_s, car_d, steer, obstacles, track,
                                     float(creep_speed))
            self.last_details = details
            if ok:
                self.active = True
                self.last_details["creep_speed"] = float(creep_speed)
                return float(creep_speed)
            if details.get("reason") not in ("predicted_wall", "predicted_gap"):
                break
        self.reset()
        return None

    def _safe(self, state, car_s, car_d, steer, obstacles, track, creep_speed):
        c = self.config
        obs = np.asarray(obstacles)
        if obs.ndim != 2 or obs.shape[1] < 6:
            return False, {"reason": "invalid_boxes"}
        # Projection uses the same resampled track as the controller.
        idx = int(round(float(car_s) / track.spacing)) % track.n
        left_wall = float(track.d_left[idx]) - self.half_width - float(car_d)
        right_wall = float(track.d_right[idx]) - self.half_width + float(car_d)
        wall_min = min(left_wall, right_wall)
        if wall_min < c.wall_min:
            return False, {"reason": "wall", "wall_min": wall_min}
        open_left = True
        open_right = True
        nearest_ahead = math.inf
        nearest_gap = math.inf
        has_blocking_ahead = False
        for o in obs:
            ahead = float(track.s_diff(o[0], car_s)) - self.half_length
            behind = -float(track.s_diff(o[1], car_s)) - self.half_length
            gap_d = max(float(o[2]) - float(car_d), float(car_d) - float(o[3])) - self.half_width
            gap_s = max(ahead, behind)
            nearest_gap = min(nearest_gap, math.hypot(max(gap_s, 0.0), max(gap_d, 0.0)))
            if 0.0 < ahead < 4.0:
                oi = int(round(0.5 * (float(o[0]) + float(o[1])) / track.spacing)) % track.n
                room_left = float(track.d_left[oi]) - self.bounds_margin - float(o[3]) - 2 * self.half_width
                room_right = float(track.d_right[oi]) - self.bounds_margin + float(o[2]) - 2 * self.half_width
                open_left &= room_left >= c.corridor_min
                open_right &= room_right >= c.corridor_min
                if gap_d < 0.0:
                    has_blocking_ahead = True
                    nearest_ahead = min(nearest_ahead, ahead)
        details = {"ahead_min": nearest_ahead, "gap_min": nearest_gap,
                   "wall_min": wall_min, "corridor_left": open_left,
                   "corridor_right": open_right}
        if (not has_blocking_ahead or nearest_ahead < c.ahead_min or
                nearest_gap < c.ahead_min or not (open_left or open_right)):
            details["reason"] = "box_or_corridor"
            return False, details
        # Constant-speed bicycle probe, recomputed at every control callback.
        x, y, yaw = (float(state[0]), float(state[1]), float(state[2]))
        base_idx, base_s, base_d = track.project(np.array([x]), np.array([y]),
                                                  np.array([idx]))
        idx = int(base_idx[0])
        base_s = float(base_s[0]); base_d = float(base_d[0])
        dt = min(0.10, c.predict_s / 3.0)
        t = 0.0
        pred_gap = nearest_gap
        pred_wall = wall_min
        while t < c.predict_s - 1e-9:
            step = min(dt, c.predict_s - t)
            x += creep_speed * math.cos(yaw) * step
            y += creep_speed * math.sin(yaw) * step
            yaw += creep_speed * math.tan(float(steer)) / self.wheelbase * step
            idx_arr, ss, dd = track.project(np.array([x]), np.array([y]), np.array([idx]))
            idx = int(idx_arr[0])
            s = (float(car_s) + float(track.s_diff(float(ss[0]), base_s))) % track.length
            d = float(car_d) + float(dd[0]) - base_d
            wall_idx = int(round(s / track.spacing)) % track.n
            wall = min(float(track.d_left[wall_idx]) - self.half_width - d,
                       float(track.d_right[wall_idx]) - self.half_width + d)
            pred_wall = min(pred_wall, wall)
            if wall < c.wall_min:
                details.update(reason="predicted_wall", predicted_wall_min=wall)
                return False, details
            for o in obs:
                ahead = float(track.s_diff(o[0] + float(o[4]) * (t + step), s)) - self.half_length
                behind = -float(track.s_diff(o[1] + float(o[4]) * (t + step), s)) - self.half_length
                od0 = float(o[2]) + float(o[5]) * (t + step)
                od1 = float(o[3]) + float(o[5]) * (t + step)
                gap_d = max(od0 - d, d - od1) - self.half_width
                gap = math.hypot(max(max(ahead, behind), 0.0), max(gap_d, 0.0))
                pred_gap = min(pred_gap, gap)
            if pred_gap < c.pred_gap_min:
                details.update(reason="predicted_gap", predicted_gap_min=pred_gap)
                return False, details
            t += step
        details["predicted_gap_min"] = pred_gap
        details["predicted_wall_min"] = pred_wall
        details["reason"] = "allowed"
        return True, details
