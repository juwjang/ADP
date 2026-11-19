"""raw 감지(프레임 단위)를 짧은 연속 확인으로 확정하는 소형 확정기 (ROS 비의존).

정적 장애물 가정: 시험 환경의 장애물은 움직이지 않으므로(스포너 설정) 속도 추정(추적 필터) 없이 위치만 연결·평균한다.
이 가정이 성립하지 않는 환경에서는 쓰지 않는다(플래그 기본 0). 인식 노드의 추적 발행(10회 측정 확정)보다 빨리,
단 연속 몇 프레임·크기·경계 조건으로 오탐을 거른 뒤에만 MPPI 입력에 합친다.
"""
import numpy as np


class RawConfirmer:
    def __init__(self, track, confirm_frames=3, assoc_dist=0.5, ttl_s=1.0,
                 min_size=0.15, max_size=0.8, bounds_margin=0.1, max_gap_s=0.1):
        self.track = track
        self.confirm_frames = confirm_frames
        self.assoc_dist = assoc_dist
        self.ttl_s = ttl_s
        self.min_size = min_size
        self.max_size = max_size
        self.bounds_margin = bounds_margin
        self.max_gap_s = max_gap_s
        self.cands = []      # dict(obs=(4,), n=int, last=t, first=t, count_t=t, done=bool)

    def _valid(self, det):
        s0, s1, dr, dl = det[:4]
        size = max(s1 - s0, dl - dr)
        if not (self.min_size <= size <= self.max_size):
            return False
        tr = self.track
        i = int(round(0.5 * (s0 + s1) / tr.spacing)) % tr.n
        dc = 0.5 * (dr + dl)
        return -(tr.d_right[i] - self.bounds_margin) <= dc <= tr.d_left[i] - self.bounds_margin

    def update(self, t, dets, car_s=None):
        """한 프레임의 감지 (N, >=4) [s_start, s_end, d_right, d_left, ...]를 넣는다."""
        tr = self.track
        dets = np.asarray(dets, dtype=float).reshape(-1, max(4, np.shape(dets)[-1] if np.ndim(dets) > 1 else 4))
        seen = set()
        for det in dets:
            if not self._valid(det):
                continue
            sc, dc = 0.5 * (det[0] + det[1]), 0.5 * (det[2] + det[3])
            best, bd = None, self.assoc_dist
            for j, c in enumerate(self.cands):
                if j in seen:
                    continue
                cs, cd = 0.5 * (c['obs'][0] + c['obs'][1]), 0.5 * (c['obs'][2] + c['obs'][3])
                dist = np.hypot(tr.s_diff(sc, cs), dc - cd)
                if dist < bd:
                    best, bd = j, dist
            if best is None:
                self.cands.append(dict(obs=det[:4].copy(), n=1, last=t, first=t, done=False))
                seen.add(len(self.cands) - 1)
            else:
                c = self.cands[best]
                if t - c['last'] > self.max_gap_s and not c['done']:
                    c['n'] = 0                      # 연속이 끊김: 다시 센다
                c['n'] += 1
                a = 1.0 / min(c['n'], 10)
                c['obs'] = (1 - a) * c['obs'] + a * det[:4]
                c['last'] = t
                if c['n'] >= self.confirm_frames:
                    c['done'] = True
                seen.add(best)
        keep = []
        for c in self.cands:
            if t - c['last'] > self.ttl_s:
                continue
            if car_s is not None and c['done'] and tr.s_diff(car_s, c['obs'][1]) > 2.0:
                continue
            if not c['done'] and t - c['last'] > self.max_gap_s and c['n'] < self.confirm_frames:
                # 확정 전 후보는 연속이 끊기면 버린다
                if t - c['last'] > 2 * self.max_gap_s:
                    continue
            keep.append(c)
        self.cands = keep

    def confirmed(self):
        out = [np.concatenate((c['obs'], [0.0, 0.0])) for c in self.cands if c['done']]
        return np.array(out).reshape(-1, 6)


def merge_obstacles(published, confirmed, track, tol_s=1.0, tol_d=0.7):
    """추적 발행 장애물 (M,6)과 확정 raw 장애물 (N,6)을 합친다. 겹치면(중심 s·d 근접) 추적 발행을 우선한다."""
    published = np.asarray(published, dtype=float).reshape(-1, 6)
    confirmed = np.asarray(confirmed, dtype=float).reshape(-1, 6)
    keep = []
    for c in confirmed:
        cs, cd = 0.5 * (c[0] + c[1]), 0.5 * (c[2] + c[3])
        dup = any(abs(track.s_diff(cs, 0.5 * (p[0] + p[1]))) < tol_s and abs(cd - 0.5 * (p[2] + p[3])) < tol_d
                  for p in published)
        if not dup:
            keep.append(c)
    if not keep:
        return published
    return np.vstack((published, np.array(keep)))
