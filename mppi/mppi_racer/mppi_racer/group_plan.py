"""지도 기반 그룹 통과 쪽 계획(grp_plan).

보이는 장애물을 s 간격으로 묶고, 묶음 안 장애물마다 좌/우 조합을 지도 정보(벽 여유, 곡률, 회전 반경,
횡이동 가능성)로 평가해 가능하고 횡이동이 가장 적은 조합을 고른다. 결과는 s에 대한 목표 d 프로파일
(차 위치 기준 상대 s)로, MPPI의 추종 비용 기준선과 PP 가이드 후보에 쓰인다. 장애물이 없으면 호출되지 않는다.
"""
import itertools
import numpy as np


class GroupPlanner:
    def __init__(self, track, half_width, rng_obs=9.0, gap=4.0, c_pref=0.25, c_min=0.10,
                 kappa_max=1.1, shift_ay=6.0, release=3.0, w_clear=2.0, kappa_hard=1.35):
        self.tr = track
        self.hw = half_width
        self.range = rng_obs
        self.gap = gap
        self.c_pref = c_pref
        self.c_min = c_min
        self.kappa_max = kappa_max
        self.shift_ay = shift_ay
        self.release = release
        self.w_clear = w_clear
        self.kappa_hard = kappa_hard
        self.mem = []          # dict(s, d, side): 묶음이 지날 때까지 통과 쪽 유지

    # ---- 지도 정보 ----
    def _walls(self, s0, s1):
        tr = self.tr
        ss = np.linspace(s0, s1, 5) % tr.length
        ids = np.round(ss / tr.spacing).astype(int) % tr.n
        return float(tr.d_left[ids].min()), float(tr.d_right[ids].min())

    def _side_target(self, ob, side):
        """(목표 d, 사용 여유) 또는 None. side=+1 왼쪽 통과, -1 오른쪽 통과."""
        s0, s1, d_lo, d_hi = ob[0], ob[1], ob[2], ob[3]
        wl, wr = self._walls(s0, s1)
        if side > 0:
            free = (wl - self.hw) - (d_hi + self.hw)   # 장애물 위쪽 가장자리와 왼쪽 벽 사이에서 차가 쓸 수 있는 폭
            if free < 2 * self.c_min:
                return None
            c = min(self.c_pref, free / 2.0)
            return d_hi + self.hw + c, c
        free = (d_lo - self.hw) + wr - self.hw
        if free < 2 * self.c_min:
            return None
        c = min(self.c_pref, free / 2.0)
        return d_lo - self.hw - c, c

    def _kappa_eff(self, s, d):
        tr = self.tr
        k = float(tr.kappa[int(round((s % tr.length) / tr.spacing)) % tr.n])
        den = 1.0 - k * d
        if den < 0.1:
            return np.inf
        return k / den

    # ---- 계획 ----
    def plan(self, car_s, car_d, v, obstacles):
        """묶음 계획. 반환 dict(knots_s(차 기준 상대 s), knots_d, sides, cost, n) 또는 None."""
        tr = self.tr
        obs = np.asarray(obstacles, dtype=float).reshape(-1, 6)
        rows = []
        for o in obs:
            a0 = float(tr.s_diff(o[0], car_s))
            a1 = float(tr.s_diff(o[1], car_s))
            if a1 <= 0.0 or a0 > self.range:
                continue
            rows.append((a0, a1, o))
        self.mem = [m for m in self.mem if float(tr.s_diff(m["s"], car_s)) > -1.0]
        if not rows:
            return None
        rows.sort(key=lambda r: r[0])
        grp = [rows[0]]
        for r in rows[1:]:
            if r[0] - grp[-1][1] <= self.gap:
                grp.append(r)
            else:
                break
        n = len(grp)
        opts = []
        for a0, a1, o in grp:
            sc = 0.5 * (o[0] + o[1])
            dc = 0.5 * (o[2] + o[3])
            fixed = None
            for m in self.mem:
                if abs(float(tr.s_diff(m["s"], sc))) < 0.6 and abs(m["d"] - dc) < 0.6:
                    fixed = m["side"]
            cand = {}
            for side in (+1, -1):
                t = self._side_target(o, side)
                if t is not None:
                    cand[side] = t
            opts.append((fixed, cand))
        best = None
        for use_fixed in (True, False):
            for combo in itertools.product((+1, -1), repeat=n):
                ok = True
                for i, side in enumerate(combo):
                    fixed, cand = opts[i]
                    if side not in cand or (use_fixed and fixed is not None and side != fixed):
                        ok = False
                        break
                if not ok:
                    continue
                cost, feasible, knots = self._evaluate(grp, combo, opts, car_d, v)
                if feasible and (best is None or cost < best[0]):
                    best = (cost, combo, knots)
            if best is not None:
                break
        if best is None:
            return None
        cost, combo, knots = best
        for (a0, a1, o), side in zip(grp, combo):
            sc = 0.5 * (o[0] + o[1])
            dc = 0.5 * (o[2] + o[3])
            hit = False
            for m in self.mem:
                if abs(float(tr.s_diff(m["s"], sc))) < 0.6 and abs(m["d"] - dc) < 0.6:
                    m["side"] = side
                    m["s"] = sc
                    m["d"] = dc
                    hit = True
            if not hit:
                self.mem.append(dict(s=sc, d=dc, side=side))
        ks = np.array([k[0] for k in knots])
        kd = np.array([k[1] for k in knots])
        return dict(knots_s=ks, knots_d=kd, sides=combo, cost=cost, n=n)

    def _evaluate(self, grp, combo, opts, car_d, v):
        v = max(float(v), 1.0)
        prev_s, prev_d = 0.0, float(car_d)
        knots = [(0.0, float(car_d))]
        cost = 0.0
        for i, ((a0, a1, o), side) in enumerate(zip(grp, combo)):
            d_t, c = opts[i][1][side]
            sc = 0.5 * (o[0] + o[1])
            ka = max(a0 - 0.3, prev_s + 0.2)
            kb = a1 + 0.3
            dd = d_t - prev_d
            ds = max(ka - prev_s, 0.2)
            k_shift = np.pi ** 2 * abs(dd) / (2.0 * ds ** 2)
            ke = abs(self._kappa_eff(sc, d_t))
            if ke > self.kappa_max:      # 목표 d에서의 경로 곡률(회전 반경)은 지도로 정해지는 하드 조건
                return np.inf, False, None
            # 횡이동이 너무 급하면(남은 거리 짧음) 불가로 버리지 않고 벌점: 늦게 알아도 가장 가능한 쪽 계획을 낸다
            pen = max(v * v * k_shift / self.shift_ay - 1.0, 0.0) + max(k_shift / self.kappa_max - 1.0, 0.0)                 + max((ke + k_shift) / self.kappa_hard - 1.0, 0.0)
            cost += abs(dd) + self.w_clear * max(self.c_pref - c, 0.0) + 2.0 * pen
            knots.append((ka, d_t))
            knots.append((kb, d_t))
            prev_s, prev_d = kb, d_t
        knots.append((prev_s + self.release, 0.0))
        out = [knots[0]]
        for s, d in knots[1:]:
            out.append((max(s, out[-1][0] + 0.05), d))
        return cost, True, out


def dref_profile(knots_s, knots_d, s_rel):
    """코사인 보간한 목표 d. s_rel은 차 기준 상대 s(배열). 범위 밖은 양 끝 값으로 고정."""
    s_rel = np.asarray(s_rel, dtype=float)
    k = np.clip(np.searchsorted(knots_s, s_rel, side="right") - 1, 0, len(knots_s) - 2)
    s0, s1 = knots_s[k], knots_s[k + 1]
    f = np.clip((s_rel - s0) / np.maximum(s1 - s0, 1e-6), 0.0, 1.0)
    w = 0.5 * (1.0 - np.cos(np.pi * f))
    return knots_d[k] + (knots_d[k + 1] - knots_d[k]) * w
