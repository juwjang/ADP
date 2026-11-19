"""계획한 가속도를 /drive의 speed 명령으로 바꾼다."""


class SteeringObserver:
    """Rate-limited servo estimate using the command active in each interval."""

    def __init__(self, rate):
        self.rate = rate
        self.angle = 0.0
        self.target = 0.0
        self.time = None

    def estimate(self, now):
        if self.time is not None:
            limit = self.rate * max(0.0, now - self.time)
            self.angle += min(max(self.target - self.angle, -limit), limit)
        self.time = now
        return self.angle

    def command(self, target, now):
        self.estimate(now)
        self.target = target


def speed_command(v, accel, k_accel, k_brake, v_max):
    """시뮬레이터(f1tenth_gym)는 speed 명령과 현재 속도의 차에 비례 게인을 곱해 가속한다.

    원하는 가속도를 내려면 speed = v + accel / 게인 으로 명령한다.
    게인은 가속과 감속에서 다르다(gym의 pid: 10*a_max/v_max, 10*a_max/|v_min|).
    """
    gain = k_accel if accel >= 0.0 else k_brake
    return min(max(v + accel / gain, 0.0), v_max)


def lowpass(prev, new, alpha):
    """1차 저역통과. alpha=1이면 필터 없음(기존 동작), 작을수록 느리게 따라간다."""
    return prev + alpha * (new - prev)


def predict_state(state, steer_est, steer_cmd, accel, delay, wheelbase, v_max, steer_max,
                  steer_rate_max):
    """상태를 delay 초 앞으로 예측한다(직전 명령이 계속 유지된다고 가정).

    반환: (예측 상태 (x, y, yaw, v), 예측 시점의 실제 조향각 추정).
    delay<=0이면 입력을 그대로 돌려준다.
    """
    import numpy as np
    from .model import rollout_with_steering_lag
    if delay <= 0.0:
        return np.asarray(state, dtype=float), steer_est
    controls = np.array([[[steer_cmd, accel]]])
    traj, steer_hist = rollout_with_steering_lag(
        np.asarray(state, dtype=float), steer_est, controls, delay, wheelbase, v_max, steer_max,
        steer_rate_max)
    return traj[0, 1], float(steer_hist[0, 0])
