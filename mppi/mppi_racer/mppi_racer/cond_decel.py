"""조건부 감속: 확정된 장애물이 있고 옆으로 겹치거나 회피 이동이 필요할 때만 속도 명령 상한을 낸다 (ROS 비의존).

출발 속도 일괄 제한이 아니다. 장애물이 없거나 이미 지나쳤으면(차 앞끝이 장애물 뒤끝을 넘음) 상한이 없다(inf).
상한 = sqrt(2·a_dec·max(L - margin, 0)) + v_min, L은 차 앞끝에서 장애물 뒤끝까지의 세로 거리.
"""
import numpy as np


def cond_decel_cap(track, car_s, car_d, obstacles, half_width, half_length,
                   a_dec=2.5, margin=0.8, vmin=1.5, lat_window=0.8):
    obstacles = np.asarray(obstacles, dtype=float).reshape(-1, 6)
    cap = np.inf
    for o in obstacles:
        ahead = float(track.s_diff(o[0], car_s)) - half_length
        if ahead <= 0.0:
            continue                                   # 나란히 있거나 이미 지나침: 해제
        gap_d = max(o[2] - car_d, car_d - o[3]) - half_width
        if gap_d >= lat_window:
            continue                                   # 옆으로 충분히 떨어져 회피 이동이 필요 없음
        cap = min(cap, float(np.sqrt(2.0 * a_dec * max(ahead - margin, 0.0)) + vmin))
    return cap
