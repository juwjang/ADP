"""mppi_racer ROS 노드: 전역 경로·차량 상태·장애물을 받아 /drive를 직접 발행한다."""
import math
import csv
import time
from dataclasses import fields

import numpy as np
import rclpy
from ackermann_msgs.msg import AckermannDriveStamped
from f110_msgs.msg import ObstacleArray, WpntArray
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node

from .command import speed_command, SteeringObserver, lowpass, predict_state
from .mppi import MPPI, MPPIParams
from .costs import shield_command
from .stuck_release import EscapeConfig, ShieldStuckRelease
from .raw_confirm import RawConfirmer, merge_obstacles
from .cond_decel import cond_decel_cap
from .track import Track


class MPPIRacer(Node):
    def __init__(self):
        super().__init__('mppi_racer')
        values = {}
        for f in fields(MPPIParams):
            values[f.name] = self.declare_parameter(f.name, f.default).value
        self.params = MPPIParams(**values)
        self.control_rate = self.declare_parameter('control_rate', 40.0).value
        # raw 감지 직접 확정(플래그 기본 0 = 기존 동작): 정적 장애물 가정(raw_confirm.py 설명 참고)
        self.raw_enabled = bool(self.declare_parameter('raw_obstacles', 0).value)
        self.raw_confirm_frames = int(self.declare_parameter('raw_confirm_frames', 3).value)
        self.raw_min_size = float(self.declare_parameter('raw_min_size', 0.08).value)
        self._raw_conf = None
        # 조건부 감속(플래그 기본 0): 확정 장애물이 있고 옆으로 겹칠 때만 속도 명령 상한(cond_decel.py)
        self.cond_decel = bool(self.declare_parameter('cond_decel', 0).value)
        self.cd_a_dec = float(self.declare_parameter('cd_a_dec', 2.5).value)
        self.cd_margin = float(self.declare_parameter('cd_margin', 0.8).value)
        self.cd_vmin = float(self.declare_parameter('cd_vmin', 1.5).value)
        self.cd_lat = float(self.declare_parameter('cd_lat', 0.8).value)
        self._pub_obs = np.zeros((0, 6))
        # f1tenth_gym의 속도 비례 게인 (sim_params: a_max 9.51, v_max 20, v_min -5)
        self.k_accel = self.declare_parameter('k_accel', 4.755).value
        self.k_brake = self.declare_parameter('k_brake', 19.02).value
        self.state_timeout = self.declare_parameter('state_timeout', 0.5).value
        self.seed = self.declare_parameter('seed', 0).value
        # 구조 변경 시험용(기본값은 기존 동작): 지연 보상 시간(s), 조향 출력 저역통과 계수(1=끔)
        self.delay_comp_s = self.declare_parameter('delay_comp_s', 0.0).value
        self.steer_lpf_alpha = self.declare_parameter('steer_lpf_alpha', 1.0).value
        self.speed_lookahead_steps = self.declare_parameter('speed_lookahead_steps', 0).value
        self.speed_lookahead_vmin = self.declare_parameter('speed_lookahead_vmin', 0.0).value
        self.start_cap_s = self.declare_parameter('start_cap_s', 0.0).value
        self.start_speed_cap = self.declare_parameter('start_speed_cap', 3.0).value
        self._first_control = None
        self.unstick_s = self.declare_parameter('unstick_s', 0.0).value
        self.unstick_hold_s = self.declare_parameter('unstick_hold_s', 2.0).value
        self.unstick_speed = self.declare_parameter('unstick_speed', 1.0).value
        self._moved_time = None
        self._unstick_until = None
        self.shield_stop_dist = self.declare_parameter('shield_stop_dist', 0.0).value
        self.shield_fix = int(self.declare_parameter('shield_fix', 0).value)
        self.shield_stuck_escape = bool(self.declare_parameter('shield_stuck_escape', 0).value)
        self._stuck_release = None
        if self.shield_stuck_escape:
            if self.shield_fix != 0:
                raise ValueError('shield_stuck_escape requires shield_fix=0')
            escape_config = EscapeConfig(
                hold_s=float(self.declare_parameter('escape_hold_s', 2.0).value),
                stationary_s_max=float(self.declare_parameter('escape_stationary_s_max', 0.05).value),
                v_max=float(self.declare_parameter('escape_v_max', 0.10).value),
                accel_min=float(self.declare_parameter('escape_accel_min', 1.0).value),
                ttc_min=float(self.declare_parameter('escape_ttc_min', 0.08).value),
                ahead_min=float(self.declare_parameter('escape_ahead_min', 0.40).value),
                corridor_min=float(self.declare_parameter('escape_corridor_min', 0.40).value),
                wall_min=float(self.declare_parameter('escape_wall_min', 0.03).value),
                pred_gap_min=float(self.declare_parameter('escape_pred_gap_min', 0.35).value),
                speed=float(self.declare_parameter('escape_speed', 0.30).value),
                speed_min=float(self.declare_parameter('escape_speed_min', 0.10).value),
                predict_s=float(self.declare_parameter('escape_predict_s', 0.10).value))
            self._stuck_release = ShieldStuckRelease(
                escape_config, self.params.half_width, self.params.half_length,
                self.params.bounds_margin, self.params.wheelbase,
                self.params.shield_speed, self.params.ttc_floor)
        self.vis_cap_scale = self.declare_parameter('vis_cap_scale', 1.0).value
        self._vis_s = self._vis_v = None
        vis_file = self.declare_parameter('vis_cap_file', '').value
        if vis_file:
            _rows = np.loadtxt(vis_file, delimiter=',', ndmin=2)
            self._vis_s, self._vis_v = _rows[:, 0], _rows[:, 1]
        self._last_accel = 0.0
        self._last_steer_out = 0.0
        # 비어 있지 않으면 매 제어 스텝을 CSV로 남긴다(진단용).
        log_path = self.declare_parameter('log_path', '').value
        self._log = open(log_path, 'w') if log_path else None
        profile_path = self.declare_parameter('profile_path', '').value
        self._profile_file = open(profile_path, 'w') if profile_path else None
        self._profile_writer = csv.writer(self._profile_file) if self._profile_file else None
        self._profile_count = 0
        if self._profile_writer:
            self._profile_writer.writerow(['t', 'prepare_ms', 'sample_ms', 'rollout_ms',
                                           'cost_ms', 'update_ms', 'final_ms', 'total_ms',
                                           'callback_ms', 'n_obs'])
        if self._log:
            self._log.write('t,x,y,yaw,v,s,d,steer,accel,speed_cmd,step_ms,min_ttc,shield,n_obs\n')

        self.mppi = None
        self._vx_ref = None
        self.pose = None            # (x, y, yaw)
        self.pose_time = None
        self.speed = None
        self.obstacles = np.zeros((0, 6))
        self._since_shift = 0.0
        self._last_step = None
        self._stats = []            # 스텝 계산 시간 (ms)
        self._shield_count = 0
        self._min_ttc = math.inf
        self._steering = SteeringObserver(self.params.steer_rate_max)

        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.create_subscription(WpntArray, '/global_waypoints_scaled', self.waypoints_cb, 10)
        self.create_subscription(PoseStamped, '/car_state/pose', self.pose_cb, 10)
        self.create_subscription(Odometry, '/car_state/odom', self.odom_cb, 10)
        self.create_subscription(ObstacleArray, '/perception/obstacles', self.obstacles_cb, 10)
        if self.raw_enabled:
            self.create_subscription(ObstacleArray, '/perception/detection/raw_obstacles', self.raw_cb, 10)
        self.create_timer(1.0 / self.control_rate, self.control_cb)
        self.create_timer(2.0, self.report_cb)
        self.get_logger().info(
            f'applied params: {self.params} | k_accel={self.k_accel} k_brake={self.k_brake} '
            f'delay_comp_s={self.delay_comp_s} steer_lpf_alpha={self.steer_lpf_alpha} '
            f'speed_lookahead_steps={self.speed_lookahead_steps} control_rate={self.control_rate}')
        self.get_logger().info(
            f'mppi_racer: K={self.params.samples}, horizon={self.params.horizon}x{self.params.dt}s, '
            f'{self.control_rate:.0f} Hz. Waiting for /global_waypoints_scaled')

    # ---- 입력 ----------------------------------------------------------
    def waypoints_cb(self, msg):
        w = msg.wpnts
        if len(w) < 4:
            return
        x = np.array([p.x_m for p in w])
        y = np.array([p.y_m for p in w])
        vx = np.array([p.vx_mps for p in w])
        d_left = np.array([p.d_left for p in w])
        d_right = np.array([p.d_right for p in w])
        if math.hypot(x[0] - x[-1], y[0] - y[-1]) < 1e-3:      # 닫힌 경로의 중복 끝점 제거
            x, y, vx, d_left, d_right = x[:-1], y[:-1], vx[:-1], d_left[:-1], d_right[:-1]
        if self._vx_ref is not None and len(vx) == len(self._vx_ref) and np.allclose(vx, self._vx_ref):
            return
        self._vx_ref = vx
        track = Track(x, y, vx, d_left, d_right)
        if self.mppi is None:
            self.mppi = MPPI(track, self.params, seed=self.seed)
        else:
            self.mppi.track = track                           # 속도 스케일만 바뀐 경우
            self.mppi._idx = None
        self.get_logger().info(
            f'track: {track.length:.2f} m, vx {track.vx.min():.2f}~{track.vx.max():.2f} m/s')

    def pose_cb(self, msg):
        q = msg.pose.orientation
        yaw = math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z))
        self.pose = (msg.pose.position.x, msg.pose.position.y, yaw)
        self.pose_time = time.monotonic()

    def odom_cb(self, msg):
        self.speed = msg.twist.twist.linear.x

    def obstacles_cb(self, msg):
        self._pub_obs = np.array(
            [[o.s_start, o.s_end, o.d_right, o.d_left,
              0.0 if o.is_static else o.vs, 0.0 if o.is_static else o.vd]
             for o in msg.obstacles]).reshape(-1, 6)
        self._refresh_obstacles()

    def raw_cb(self, msg):
        if self.mppi is None:
            return
        if self._raw_conf is None or self._raw_conf.track is not self.mppi.track:
            self._raw_conf = RawConfirmer(self.mppi.track, confirm_frames=self.raw_confirm_frames,
                                          min_size=self.raw_min_size)
        dets = np.array([[o.s_start, o.s_end, o.d_right, o.d_left] for o in msg.obstacles]).reshape(-1, 4)
        car_s = None if self.mppi._idx is None else self.mppi._idx * self.mppi.track.spacing
        self._raw_conf.update(time.monotonic(), dets, car_s=car_s)
        self._refresh_obstacles()

    def _refresh_obstacles(self):
        if self.raw_enabled and self._raw_conf is not None and self.mppi is not None:
            self.obstacles = merge_obstacles(self._pub_obs, self._raw_conf.confirmed(), self.mppi.track)
        else:
            self.obstacles = self._pub_obs

    # ---- 제어 ----------------------------------------------------------
    def publish(self, steer, speed):
        msg = AckermannDriveStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'base_link'
        msg.drive.steering_angle = float(steer)
        msg.drive.speed = float(speed)
        self.drive_pub.publish(msg)
        # Also track safety-stop commands, not only normal optimizer output.
        self._steering.command(float(steer), time.monotonic())

    def control_cb(self):
        if self.mppi is None or self.pose is None or self.speed is None:
            return
        now = time.monotonic()
        if now - self.pose_time > self.state_timeout:
            self.publish(0.0, 0.0)                            # 상태가 끊기면 정지
            return
        # 제어 주기가 dt보다 짧으므로, dt만큼 시간이 지났을 때만 계획을 한 칸 민다.
        if self._last_step is not None:
            self._since_shift += now - self._last_step
        self._last_step = now
        shift = self._since_shift >= self.params.dt
        if shift:
            self._since_shift = min(self._since_shift - self.params.dt, self.params.dt)
        v = max(self.speed, 0.0)
        state = np.array([self.pose[0], self.pose[1], self.pose[2], v])
        profile = {} if self._profile_writer else None
        steer_now = self._steering.estimate(now)
        plan_state = state
        if self.delay_comp_s > 0.0:
            p = self.params
            plan_state, steer_now = predict_state(
                state, steer_now, self._steering.target, self._last_accel, self.delay_comp_s,
                p.wheelbase, p.v_max, p.steer_max, p.steer_rate_max)
        res = self.mppi.step(plan_state, self.obstacles, shift=shift, steer_now=steer_now,
                             profile=profile)
        steer_out = lowpass(self._last_steer_out, res.steer, self.steer_lpf_alpha)
        steer_out = max(-self.params.steer_max, min(self.params.steer_max, steer_out))
        self._last_steer_out = steer_out
        self._last_accel = res.accel
        if res.shield:
            _nl = min(max(int(self.speed_lookahead_steps), 1), len(res.traj) - 1)
            speed = shield_command(v, max(float(res.traj[_nl, 3]), 0.0), self.params.shield_speed, self.shield_fix)
            _escape_forced_stop = False
            if self.shield_stop_dist > 0.0 and len(self.obstacles):
                # 쉴드 중에도 앞 shield_stop_dist m 안에 같은 차선 장애물이 있으면 완전 정지한다(기본 꺼짐).
                tr = self.mppi.track
                _, s_car, d_car = tr.project(state[:1], state[1:2], np.array([self.mppi._idx]))
                hw = self.params.half_width
                for o in self.obstacles:
                    ahead = float(tr.s_diff(o[0], s_car[0])) - self.params.half_length
                    overlap = (d_car[0] + hw > o[2]) and (d_car[0] - hw < o[3])
                    if overlap and 0.0 < ahead < self.shield_stop_dist:
                        speed = 0.0
                        _escape_forced_stop = True
                        break
            if self._stuck_release is not None:
                tr = self.mppi.track
                hint = self.mppi._idx
                if hint is None:
                    hint = tr.locate(state[0], state[1])
                _, s_car, d_car = tr.project(state[:1], state[1:2], np.array([hint]))
                creep = self._stuck_release.consider(
                    now, state, float(s_car[0]), float(d_car[0]), steer_out,
                    res.shield, res.accel, res.min_ttc, self.obstacles, tr,
                    forced_stop=_escape_forced_stop)
                if creep is not None:
                    speed = max(speed, creep)
            self._shield_count += 1
        else:
            if self._stuck_release is not None:
                self._stuck_release.reset()
            speed = speed_command(v, res.accel, self.k_accel, self.k_brake, self.params.v_max)
            n = int(self.speed_lookahead_steps)
            if n > 0:
                # 계획 속도(n 스텝 앞)를 직접 명령한다. 측정 속도가 지연으로 뒤처져도 명령이 낮게 붙지 않는다.
                n = min(n, len(res.traj) - 1)
                cmd_accel = speed
                speed = min(max(float(res.traj[n, 3]), 0.0), self.params.v_max)
                if v < self.speed_lookahead_vmin:
                    # 정지 부근에서는 계획이 곧 멈춘다고 해도(뒤쪽 감속) 첫 가속도로 출발할 수 있게 한다(교착 방지).
                    speed = max(speed, cmd_accel)
            if self.params.speed_profile_mode > 0:
                # 장애물이 없으면 프로파일 속도를 그대로, 있으면 MPPI 계획 기반 속도와의 최솟값(회피·감속 유지).
                prof = min(self.mppi.speed_profile(self.mppi._idx), self.params.v_max)
                # 명령이 현재 속도에서 너무 멀어지면 시뮬레이터가 최대 가속·제동으로 반응하므로 변화량을 제한한다.
                lo = v - 1.25 * self.params.profile_a_dec / self.k_brake
                hi = v + self.params.profile_a_acc / self.k_accel
                prof = min(max(prof, lo), hi)
                if len(self.obstacles) or (self.params.speed_profile_mode == 2
                                           and res.accel < -self.params.profile_brake_thr):
                    # 모드 2: MPPI가 크게 감속을 요구하면(미끄러짐·이탈 등) 그 감속을 우선한다.
                    speed = min(prof, speed)
                else:
                    speed = prof
        if self.cond_decel and len(self.obstacles):
            _tr2 = self.mppi.track
            _, _s2, _d2 = _tr2.project(state[:1], state[1:2], np.array([self.mppi._idx]))
            speed = min(speed, cond_decel_cap(_tr2, float(_s2[0]), float(_d2[0]), self.obstacles,
                                              self.params.half_width, self.params.half_length,
                                              self.cd_a_dec, self.cd_margin, self.cd_vmin, self.cd_lat))
        if self._vis_s is not None:
            # 가려짐 전제 속도 상한(기본 꺼짐): 경로 위치별로 인식이 장애물을 처음 알려 주는 거리의 하위 분위에서
            # 멈출 수 있는 속도로 제한한다(파일은 s,vcap 행). 인식 결과가 아니라 지도 위치만 쓴다.
            _tr = self.mppi.track
            _, _s_car, _ = _tr.project(state[:1], state[1:2], np.array([self.mppi._idx]))
            speed = min(speed, float(np.interp(_s_car[0], self._vis_s, self._vis_v, period=_tr.length)) * self.vis_cap_scale)
        if self.unstick_s > 0.0:
            # 교착 해소: 정지 상태가 unstick_s 이상 이어지면 unstick_hold_s 동안 저속 전진 명령을 낸다(기본 꺼짐).
            if v > 0.1 or self._moved_time is None:
                self._moved_time = now
            if now - self._moved_time > self.unstick_s and self._unstick_until is None:
                self._unstick_until = now + self.unstick_hold_s
            if self._unstick_until is not None:
                if now < self._unstick_until:
                    speed = max(speed, self.unstick_speed)
                else:
                    self._unstick_until = None
                    self._moved_time = now
        if self.start_cap_s > 0.0:
            # 정지 출발 직후 일정 시간은 속도를 제한해 가려진 장애물에 반응할 거리를 확보한다(기본 꺼짐).
            if self._first_control is None:
                self._first_control = now
            if now - self._first_control < self.start_cap_s:
                speed = min(speed, self.start_speed_cap)
        if self._stuck_release is not None:
            self._stuck_release.observe(speed)
        self.publish(steer_out, speed)
        step_ms = (time.monotonic() - now) * 1000.0
        self._stats.append(step_ms)
        if self._log:
            tr = self.mppi.track
            _, s, d = tr.project(state[:1], state[1:2], np.array([self.mppi._idx]))
            self._log.write(
                f'{now:.4f},{state[0]:.4f},{state[1]:.4f},{state[2]:.4f},{v:.4f},{s[0]:.4f},{d[0]:.4f},'
                f'{steer_out:.4f},{res.accel:.4f},{speed:.4f},{step_ms:.2f},{res.min_ttc:.3f},'
                f'{int(res.shield)},{len(self.obstacles)}\n')
            self._log.flush()
        if self._profile_writer:
            self._profile_writer.writerow([now] + [profile[k] for k in (
                'prepare_ms', 'sample_ms', 'rollout_ms', 'cost_ms', 'update_ms',
                'final_ms', 'total_ms')] + [(time.monotonic()-now)*1000, len(self.obstacles)])
            self._profile_count += 1
            if self._profile_count % 40 == 0:
                self._profile_file.flush()
        self._min_ttc = min(self._min_ttc, res.min_ttc)

    def report_cb(self):
        if not self._stats:
            return
        t = np.array(self._stats)
        period_ms = 1000.0 / self.control_rate
        self.get_logger().info(
            f'step ms mean {t.mean():.1f} max {t.max():.1f} over-period {int((t > period_ms).sum())}/{len(t)} '
            f'| v {self.speed:.2f} | min_ttc {self._min_ttc:.2f} | shield {self._shield_count} '
            f'| obstacles {len(self.obstacles)}')
        self._stats = []
        self._shield_count = 0
        self._min_ttc = math.inf

    def destroy_node(self):
        for handle in (self._log, self._profile_file):
            if handle:
                handle.close()
        return super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = MPPIRacer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
