"""전역 경로(닫힌 트랙)를 일정 간격으로 재샘플링하고 프레넷 좌표로 변환한다."""
import numpy as np
from scipy.interpolate import CubicSpline


class Track:
    """닫힌 기준 경로. +d는 진행 방향의 왼쪽이다."""

    def __init__(self, x, y, vx, d_left, d_right, spacing=0.1):
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        xc = np.append(x, x[0])
        yc = np.append(y, y[0])
        s_raw = np.concatenate(([0.0], np.cumsum(np.hypot(np.diff(xc), np.diff(yc)))))
        self.length = float(s_raw[-1])
        self.n = int(round(self.length / spacing))
        self.spacing = self.length / self.n
        self.s = np.arange(self.n) * self.spacing

        def interp(values):
            v = np.asarray(values, dtype=float)
            return np.interp(self.s, s_raw, np.append(v, v[0]))

        # x, y는 주기 3차 스플라인으로 보간해 헤딩과 곡률을 매끄럽게 얻는다.
        spline = CubicSpline(s_raw, np.column_stack((xc, yc)), bc_type='periodic')
        pos = spline(self.s)
        d1 = spline(self.s, 1)
        d2 = spline(self.s, 2)
        self.x = pos[:, 0]
        self.y = pos[:, 1]
        self.vx = interp(vx)
        self.d_left = interp(d_left)
        self.d_right = interp(d_right)
        self.psi = np.arctan2(d1[:, 1], d1[:, 0])
        self.cos_psi = np.cos(self.psi)
        self.sin_psi = np.sin(self.psi)
        # The default search window is reused for every candidate at every step.
        self._default_candidates = (np.arange(self.n)[:, None] + np.arange(-3, 13)) % self.n
        self.kappa = (d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]) / np.hypot(d1[:, 0], d1[:, 1]) ** 3

    def locate(self, x, y):
        """전체 탐색으로 가장 가까운 기준점의 인덱스를 구한다(초기화용)."""
        return int(np.argmin((self.x - x) ** 2 + (self.y - y) ** 2))

    def project(self, x, y, hint, back=3, fwd=12):
        """hint 인덱스 주변 윈도우에서 가장 가까운 점을 찾아 (idx, s, d)를 구한다."""
        hint = np.asarray(hint)
        if back == 3 and fwd == 12:
            cand = self._default_candidates[hint % self.n]
        else:
            offsets = np.arange(-back, fwd + 1)
            cand = (hint[..., None] + offsets) % self.n
        d2 = (self.x[cand] - x[..., None]) ** 2 + (self.y[cand] - y[..., None]) ** 2
        best = np.argmin(d2, axis=-1)
        idx = np.take_along_axis(cand, best[..., None], axis=-1)[..., 0]
        dx = x - self.x[idx]
        dy = y - self.y[idx]
        along = dx * self.cos_psi[idx] + dy * self.sin_psi[idx]
        s = (idx * self.spacing + along) % self.length
        d = -dx * self.sin_psi[idx] + dy * self.cos_psi[idx]
        return idx, s, d

    def s_diff(self, a, b):
        """랩 경계를 고려한 부호 있는 최단 거리 a - b."""
        half = 0.5 * self.length
        return (a - b + half) % self.length - half


def _segments_cross(p, q, a, b):
    """선분 p-q 와 선분들 a[j]-b[j] 의 교차 여부 (j별 bool)."""
    def cross(u, v):
        return u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
    r = q - p
    s = b - a
    denom = cross(r, s)
    ap = a - p
    with np.errstate(divide='ignore', invalid='ignore'):
        t = cross(ap, s) / denom
        u = cross(ap, r) / denom
    return (np.abs(denom) > 1e-12) & (t >= 0.0) & (t <= 1.0) & (u >= 0.0) & (u <= 1.0)


def sight_distance(track, max_dist=15.0, k_step=5):
    """기준 경로 각 점에서 앞쪽 경로 점이 트랙 경계에 가려지지 않고 보이는 거리 (m, 경로를 따라 잰 값).

    지도(전역 경로의 d_left, d_right)만 쓰는 기하 계산이라 장애물 정보는 필요 없다.
    """
    n = track.n
    nl = np.column_stack((-track.sin_psi, track.cos_psi))
    pts = np.column_stack((track.x, track.y))
    left = pts + nl * track.d_left[:, None]
    right = pts - nl * track.d_right[:, None]
    kmax = int(max_dist / track.spacing)
    out = np.full(n, float(max_dist))
    for i in range(n):
        p = pts[i]
        blocked_at = kmax
        for k in range(k_step, kmax + 1, k_step):
            ids = (i + np.arange(0, k + 2)) % n
            q = pts[(i + k) % n]
            ls, le = left[ids[:-1]], left[ids[1:]]
            rs, re_ = right[ids[:-1]], right[ids[1:]]
            if _segments_cross(p, q, ls, le).any() or _segments_cross(p, q, rs, re_).any():
                blocked_at = k - k_step
                break
        out[i] = blocked_at * track.spacing
    return out
