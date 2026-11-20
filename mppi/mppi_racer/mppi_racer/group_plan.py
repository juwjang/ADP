"""지도 기반 그룹 통과 쪽 계획(grp_plan).

보이는 장애물을 s 간격으로 묶고, 묶음 안 장애물마다 좌/우 조합을 지도 정보(벽 여유, 곡률, 회전 반경,
횡이동 가능성)로 평가해 가능하고 횡이동이 가장 적은 조합을 고른다. 결과는 s에 대한 목표 d 프로파일
(차 위치 기준 상대 s)로, MPPI의 추종 비용 기준선과 PP 가이드 후보에 쓰인다. 장애물이 없으면 호출되지 않는다.
"""
import itertools
import numpy as np


class GroupPlanner:
    def __init__(self, track, half_width, rng_obs=9.0, gap=4.0, c_pref=0.25, c_min=0.10,
                 kappa_max=1.1, shift_ay=6.0, release=3.0, w_clear=2.0, kappa_hard=1.35,
                 clip=0, clip_margin=0.10, switch=0, switch_margin=2.0, switch_lock=0.0,
                 clearance=0, cl_thr=0.15, cl_w=1000.0):
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
        self.clip = int(clip)            # 1이면 전환 경로 전체가 벽 여유 안에 들도록 전환 시작을 늦추고, 안 되는 조합은 탈락
        self.clip_margin = clip_margin
        self.switch = int(switch)        # 1이면 기억된 쪽을 유지하는 조합이 기억을 푼 조합보다 switch_margin 넘게 비쌀 때 기억을 버린다
        self.switch_margin = switch_margin
        self.clearance = int(clearance)  # 1이면 조합 비용에 전환 경로 전체의 최소 여유(장애물·벽) 부족분을 지배적으로 더한다
        self.cl_thr = cl_thr             # m, 이 값보다 작은 최소 여유에 벌점
        self.cl_w = cl_w                 # 여유 부족 1 m당 비용(횡이동·급함 벌점보다 크게)
        self.switch_lock = switch_lock   # m, 기억된 쪽의 장애물이 이 거리 안(차 기준 앞쪽 거리)이면 쪽 전환 금지
        self.w_viol = 6.0                # clip=2: 벽/여유 위반 1 m당 비용(전환 총 횡이동 1 m와 같은 단위)
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

    def _limits(self, s_abs):
        """절대 s 배열 -> (하한 d, 상한 d): 차 중심이 벽 접촉 한계에서 clip_margin 떨어진 범위."""
        tr = self.tr
        ids = np.round((np.asarray(s_abs) % tr.length) / tr.spacing).astype(int) % tr.n
        return (-tr.d_right[ids] + self.hw + self.clip_margin, tr.d_left[ids] - self.hw - self.clip_margin)

    def _ramp_fits(self, car_s, s_a, d_a, s_b, d_b, s_start):
        """구간 [s_a, s_b]에서 s_start까지 d_a를 유지하다 s_b에 d_b로 코사인 전환하는 경로가 벽 여유 안인가."""
        xs = np.arange(s_a, s_b + 1e-9, 0.1)
        w = np.clip((xs - s_start) / max(s_b - s_start, 1e-6), 0.0, 1.0)
        d = d_a + (d_b - d_a) * 0.5 * (1.0 - np.cos(np.pi * w))
        lo, hi = self._limits(car_s + xs)
        mk = xs >= s_start - 1e-9          # 현재 d 유지 구간은 이미 차가 있는 위치라 검사하지 않음
        return bool(np.all(d[mk] <= hi[mk] + 1e-9) and np.all(d[mk] >= lo[mk] - 1e-9))

    def _ramp_over(self, car_s, s_a, d_a, s_b, d_b, s_start):
        """_ramp_fits의 연속판: 전환 경로가 벽 여유 밖으로 나간 최대 길이(m). 0이면 안."""
        xs = np.arange(s_a, s_b + 1e-9, 0.1)
        w = np.clip((xs - s_start) / max(s_b - s_start, 1e-6), 0.0, 1.0)
        d = d_a + (d_b - d_a) * 0.5 * (1.0 - np.cos(np.pi * w))
        lo, hi = self._limits(car_s + xs)
        mk = xs >= s_start - 1e-9
        if not mk.any():
            return 0.0
        return float(max(np.max(d[mk] - hi[mk]), np.max(lo[mk] - d[mk]), 0.0))

    def _kappa_eff(self, s, d):
        tr = self.tr
        k = float(tr.kappa[int(round((s % tr.length) / tr.spacing)) % tr.n])
        den = 1.0 - k * d
        if den < 0.1:
            return np.inf
        return k / den

    # ---- 계획 ----
    def plan(self, car_s, car_d, v, obstacles):
        self._car_s = float(car_s)
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
        def search(use_fixed):
            bst = None
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
                if feasible and (bst is None or cost < bst[0]):
                    bst = (cost, combo, knots)
            return bst

        best = search(True)
        has_fixed = any(o_[0] is not None for o_ in opts)
        near_fixed = any(o_[0] is not None and g_[0] < self.switch_lock for o_, g_ in zip(opts, grp))
        if best is None or (self.switch and has_fixed and not near_fixed):
            free = search(False)
            if best is None or (free is not None and free[0] + self.switch_margin < best[0]):
                best = free
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
            s_start = prev_s
            viol = 0.0
            if self.clip:
                xs_t = np.linspace(ka, kb, 7)
                lo_t, hi_t = self._limits(self._car_s + xs_t)
                # 목표 d를 벽 여유 안으로 당기되, 장애물과의 최소 여유(c_min)가 남아야 한다
                if side > 0:
                    d_t = min(d_t, float(hi_t.min()))
                    short = (o[3] + self.hw + self.c_min) - d_t
                else:
                    d_t = max(d_t, float(lo_t.max()))
                    short = d_t - (o[2] - self.hw - self.c_min)
                if short > 1e-9:
                    if self.clip < 2:
                        return np.inf, False, None
                    viol += short               # 소프트: 탈락시키지 않고 여유 부족만큼 벌점
                dd = d_t - prev_d
                # 전환 시작을 늦춰 가며 전환 경로 전체가 벽 여유 안에 드는 가장 이른 시작을 찾는다
                # clip=2는 횡가속 한계가 허용하는 짧은 전환 길이(ds_min)보다 늦추지 않는다
                x_max = ka - 0.05
                if self.clip >= 2:
                    ds_min = np.sqrt(np.pi ** 2 * abs(dd) * v * v / (2.0 * self.shift_ay))
                    x_max = min(x_max, ka - max(ds_min, 0.3))
                s_start = None
                x = prev_s
                while x <= x_max:
                    if self._ramp_fits(self._car_s, prev_s, prev_d, ka, d_t, x):
                        s_start = x
                        break
                    x += 0.25
                if s_start is None:
                    if self.clip < 2:
                        return np.inf, False, None
                    # 소프트: 지연 없이(또는 허용 한도까지) 가장 덜 벗어나는 시작을 고른다
                    best_x, best_o = prev_s, None
                    x = prev_s
                    while x <= max(x_max, prev_s):
                        ov = self._ramp_over(self._car_s, prev_s, prev_d, ka, d_t, x)
                        if best_o is None or ov < best_o - 1e-9:
                            best_x, best_o = x, ov
                        x += 0.25
                    s_start = best_x
                    viol += best_o if best_o is not None else 0.0
            ds = max(ka - s_start, 0.2)
            k_shift = np.pi ** 2 * abs(dd) / (2.0 * ds ** 2)
            ke = abs(self._kappa_eff(sc, d_t))
            if ke > self.kappa_max:      # 목표 d에서의 경로 곡률(회전 반경)은 지도로 정해지는 하드 조건
                return np.inf, False, None
            # 횡이동이 너무 급하면(남은 거리 짧음) 불가로 버리지 않고 벌점: 늦게 알아도 가장 가능한 쪽 계획을 낸다
            pen = max(v * v * k_shift / self.shift_ay - 1.0, 0.0) + max(k_shift / self.kappa_max - 1.0, 0.0)                 + max((ke + k_shift) / self.kappa_hard - 1.0, 0.0)
            cost += abs(dd) + self.w_clear * max(self.c_pref - c, 0.0) + 2.0 * pen + self.w_viol * viol
            if s_start > prev_s + 1e-6:
                knots.append((s_start, prev_d))     # 벽 돌출을 지날 때까지 현재 d 유지 후 전환
            knots.append((ka, d_t))
            knots.append((kb, d_t))
            prev_s, prev_d = kb, d_t
        knots.append((prev_s + self.release, 0.0))
        out = [knots[0]]
        for s, d in knots[1:]:
            out.append((max(s, out[-1][0] + 0.05), d))
        if self.clearance:
            ks = np.array([k[0] for k in out]); kd = np.array([k[1] for k in out])
            sg = np.arange(0.0, max(prev_s, 0.1), 0.1)
            dr = dref_profile(ks, kd, sg)
            lo, hi = self._limits(self._car_s + sg)       # clip_margin 포함 한계
            m = self.clip_margin
            wall_cl = np.minimum((hi + m) - dr, dr - (lo - m))   # 차 몸체와 벽 사이 거리(접촉 한계 기준)
            c_obs = min(opts[i][1][side][1] for i, side in enumerate(combo))
            mc = min(float(wall_cl.min()), c_obs)
            cost += self.cl_w * max(self.cl_thr - mc, 0.0)
        return cost, True, out


def dref_profile(knots_s, knots_d, s_rel):
    """코사인 보간한 목표 d. s_rel은 차 기준 상대 s(배열). 범위 밖은 양 끝 값으로 고정."""
    s_rel = np.asarray(s_rel, dtype=float)
    k = np.clip(np.searchsorted(knots_s, s_rel, side="right") - 1, 0, len(knots_s) - 2)
    s0, s1 = knots_s[k], knots_s[k + 1]
    f = np.clip((s_rel - s0) / np.maximum(s1 - s0, 1e-6), 0.0, 1.0)
    w = 0.5 * (1.0 - np.cos(np.pi * f))
    return knots_d[k] + (knots_d[k + 1] - knots_d[k]) * w
