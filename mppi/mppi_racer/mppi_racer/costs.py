"""MPPI 비용 항목. 모든 함수는 numpy 배열을 받아 같은 모양의 비용을 돌려준다.

장애물 배열은 (M, 6) = [s_start, s_end, d_right, d_left, vs, vd] (f110_msgs/Obstacle과 같은 뜻).
"""
import numpy as np


def tracking_cost(d, heading_err, w_d, w_heading):
    """횡오차와 헤딩 오차의 제곱 합."""
    return w_d * d ** 2 + w_heading * heading_err ** 2


def speed_cost(v, v_ref, kappa, w_speed, a_lat_max, w_alat, w_under=None, vref_gain=1.0):
    """기준 속도 오차 + 횡가속도(v^2·|kappa|) 한계 초과분.

    w_under가 주어지면 v가 목표보다 낮을 때(미달)의 가중치로 쓴다(비대칭, 기본 None = 대칭, 기존 동작).
    vref_gain은 속도 항의 목표만 배율로 올린다(a_lat 항은 원래 기준 속도를 그대로 쓴다, 기본 1.0).
    """
    over = np.maximum(v ** 2 * np.abs(kappa) - a_lat_max, 0.0)
    target = v_ref * vref_gain
    if w_under is None:
        w = w_speed
    else:
        w = np.where(v < target, w_under, w_speed)
    return w * (v - target) ** 2 + w_alat * over ** 2


def bounds_cost(d, d_left, d_right, half_width, margin, w):
    """차폭 절반 + 여유를 뺀 트랙 범위를 벗어난 거리의 제곱."""
    over_left = np.maximum(d - (d_left - half_width - margin), 0.0)
    over_right = np.maximum(-d - (d_right - half_width - margin), 0.0)
    return w * (over_left + over_right) ** 2


def wall_hit_cost(d, d_left, d_right, half_width, margin, w):
    """차체가 지도 경계를 실제로 넘는(여유 margin을 다 쓴) 스텝에 고정 페널티 w."""
    over_left = d - (d_left - half_width - margin)
    over_right = -d - (d_right - half_width - margin)
    hit = (over_left > margin) | (over_right > margin)
    return w * hit


def obstacle_gaps(s, d, t, obstacles, track, half_width, half_length):
    """차량 위치 (s, d)와 시각 t에서 등속 예측한 장애물 사이의 여유 거리.

    반환 (모양은 s.shape + (M,)):
      ahead  : 차 앞끝에서 장애물 뒤끝까지 (양수면 장애물이 앞에 있음)
      behind : 장애물 앞끝에서 차 뒤끝까지 (양수면 이미 지나침)
      gap_d  : 옆 방향 여유 (음수면 옆으로 겹침)
    ahead와 behind가 모두 음수면 세로로 겹친 상태다.
    """
    s = s[..., None]
    d = d[..., None]
    t = t[..., None]
    s_shift = obstacles[:, 4] * t
    d_shift = obstacles[:, 5] * t
    ahead = track.s_diff(obstacles[:, 0] + s_shift, s) - half_length
    behind = -track.s_diff(obstacles[:, 1] + s_shift, s) - half_length
    gap_d = np.maximum(obstacles[:, 2] + d_shift - d, d - (obstacles[:, 3] + d_shift)) - half_width
    return ahead, behind, gap_d


def obstacle_cost(ahead, behind, gap_d, margin_s, margin_d, w_collision, w_near, inflate=0.0):
    """충돌은 큰 고정 페널티, 근접은 옆 여유가 줄수록 커지는 장벽. 장애물에 대해 합한다."""
    gap_s = np.maximum(ahead, behind)
    collision = (gap_s < inflate) & (gap_d < inflate)   # inflate>0이면 장애물 박스를 사방으로 키운 충돌 판정
    proximity = np.clip(1.0 - gap_d / margin_d, 0.0, 1.0)
    cost = np.where(gap_s < margin_s, w_near * proximity ** 2, 0.0) + w_collision * collision
    return cost.sum(axis=-1)


def time_to_collision(ahead, gap_d, v, obstacle_vs, behind=None):
    """같은 차선(옆으로 겹침)의 앞 장애물에 대한 TTC의 최솟값. 접근하지 않으면 inf."""
    closing = v[..., None] - obstacle_vs
    valid = (ahead > 0.0) & (gap_d < 0.0) & (closing > 1e-3)
    ttc = np.where(valid, ahead / np.where(valid, closing, 1.0), np.inf)
    if behind is not None:
        # Already overlapping is a collision now, even at zero closing speed.
        # The rear gap distinguishes overlap from an obstacle already passed.
        overlap = (ahead <= 0.0) & (behind <= 0.0) & (gap_d < 0.0)
        ttc = np.where(overlap, 0.0, ttc)
    return ttc.min(axis=-1, initial=np.inf)


def ttc_cost(ttc, threshold, w):
    """TTC가 임계값 미만이면 0에 가까울수록 w까지 커진다."""
    return w * np.clip(1.0 - ttc / threshold, 0.0, 1.0) ** 2


def smoothness_cost(controls, prev_u, w_dsteer, w_daccel):
    """직전 입력 대비, 그리고 호라이즌 안에서의 입력 변화량 제곱 합. (K, T, 2) -> (K,)"""
    first = controls[:, :1] - prev_u
    diff = np.concatenate((first, np.diff(controls, axis=1)), axis=1)
    return (w_dsteer * diff[..., 0] ** 2 + w_daccel * diff[..., 1] ** 2).sum(axis=1)


def shield_speed(speed_cmd, min_ttc, ttc_floor, safe_speed):
    """비상 감속 보호: 최소 TTC가 하한 미만이면 속도 명령을 safe_speed 이하로 낮춘다."""
    if min_ttc < ttc_floor:
        return min(speed_cmd, safe_speed)
    return speed_cmd


def shield_command(v, v_plan, safe_speed, fix=0):
    """쉴드 발동 중 속도 명령. fix=0: min(v, safe)(기존, v=0이면 출발 불가). fix=1: 현재 속도 v가 safe 미만이면
    계획 속도 v_plan까지(최대 safe) 허용해 정지·저속에서 출발할 수 있게 한다. v>=safe이거나 v_plan<=v면 기존과 같다."""
    if fix and v < safe_speed:
        return min(safe_speed, max(v, v_plan))
    return min(v, safe_speed)


def lateral_accel_cost(v, steer, wheelbase, a_lat_max, w):
    """차량 자체의 횡가속도 v^2·tan(steer)/L 이 한계를 넘는 양의 제곱 (타이어 한계의 근사)."""
    over = np.maximum(v ** 2 * np.abs(np.tan(steer)) / wheelbase - a_lat_max, 0.0)
    return w * over ** 2


def side_targets(obstacles, track, half_width, margin_d, extra, bounds_margin, sign=None):
    """장애물마다 여유가 큰 쪽을 통과 쪽으로 정하고 목표 횡위치 d (M,)를 돌려준다(트랙 경계 안으로 제한).

    장애물 배열 (M, 6): d_right=obstacles[:, 2](아래쪽 가장자리), d_left=obstacles[:, 3](위쪽 가장자리).
    """
    mid = 0.5 * (obstacles[:, 0] + obstacles[:, 1])
    idx = np.round(mid / track.spacing).astype(int) % track.n
    room_left = (track.d_left[idx] - bounds_margin) - obstacles[:, 3]
    room_right = obstacles[:, 2] + (track.d_right[idx] - bounds_margin)
    off = half_width + margin_d + extra
    if sign is None:
        go_left = room_left >= room_right
    else:
        go_left = np.asarray(sign) > 0
    target = np.where(go_left, obstacles[:, 3] + off, obstacles[:, 2] - off)
    lo = -(track.d_right[idx] - half_width - bounds_margin)
    hi = track.d_left[idx] - half_width - bounds_margin
    return np.clip(target, lo, hi)


def side_pref_cost(ahead, behind, d, targets, window, w, lo=None, hi=None):
    """아직 지나치지 않았고 앞 window m 안에 있는 가장 가까운 장애물의 통과 목표 d로 끌어당기는 비용.

    반환: (비용, 활성 여부). 활성이 아니면(장애물 없음·멀리·이미 지나침) 비용 0.
    """
    active = (ahead < window) & (behind < 0.0)
    a = np.where(active, ahead, np.inf)
    j = np.argmin(a, axis=-1)
    has = np.isfinite(np.take_along_axis(a, j[..., None], axis=-1)[..., 0])
    tgt = targets[j]
    if lo is not None:
        tgt = np.clip(tgt, lo, hi)     # 차가 지금 있는 위치의 트랙 경계 안으로 목표를 제한
    cost = w * (d - tgt) ** 2 * has
    return cost, has


def side_room(obstacles, track, bounds_margin):
    """장애물마다 (왼쪽 여유, 오른쪽 여유): 트랙 경계(여유 포함)와 장애물 가장자리 사이 폭."""
    mid = 0.5 * (obstacles[:, 0] + obstacles[:, 1])
    idx = np.round(mid / track.spacing).astype(int) % track.n
    return (track.d_left[idx] - bounds_margin) - obstacles[:, 3], obstacles[:, 2] + (track.d_right[idx] - bounds_margin)
