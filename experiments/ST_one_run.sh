#!/usr/bin/env bash

# Run one scale-0.7 trial in the existing PC1 container. This helper is copied
# into the container for the authorized batch; it does not edit simulator code.
name="$1"
seed="$2"
obs="$3"
host_load="$4"
case "$name" in
  ST_*) tag="$name" ;;
  *) tag="ST_${name}" ;;
esac
d="$HOME/m26/$tag"
n=1
while [ -e "$d" ]; do
  n=$((n + 1))
  d="$HOME/m26/${tag}_r$n"
done
base="$(basename "$d")"
log="$HOME/m26_${base}.log"

before="$(ps -eo pid= | tr '\n' ' ')"
start="$(date +%s%N)"

B="bounds_margin=0.25,steer_lpf_alpha=0.5,speed_lookahead_steps=6,k_accel=2.5,k_brake=6.0,pp_fraction=0.25,a_lat_max=8.0,w_speed=2.0,exploration_fraction=0.2,shield_speed=1.5,speed_lookahead_vmin=1.0,w_near=30.0,obs_margin_d=0.4,w_d=30.0,yaw_understeer=0.0,model_st_substeps=2,sigma_curv_gain=3.0,sigma_steer=0.08,sigma_accel=1.0"
R="$B,raw_obstacles=1,raw_confirm_frames=3,raw_min_size=0.0,grp_plan=1,grp_clearance=1,grp_switch=1,grp_switch_lock=3.0,shield_stuck_escape=1"
source /opt/ros/humble/setup.bash >/dev/null 2>&1
source "$HOME/forza_ws/race_stack/install/setup.bash" >/dev/null 2>&1
export ROS_DOMAIN_ID=37

bash "$HOME/EV2_run.sh" "$base" --controller MPPI --scale 0.7 --obstacles "$obs" --laps 10 --seconds 180 --seed "$seed" --pose-source pf \
  --pkg /tmp/pkg_h4_stuck --k-brake 7.0 --k-accel 2.4 --mppi-params "$R" > "$log" 2>&1
run_rc=$?

mkdir -p "$d"
printf '%s\n' "$host_load" > "$d/host_cpu_load_start.txt"
(cd /tmp/pkg_h4_stuck/mppi_racer && md5sum mppi.py node.py group_plan.py command.py) > "$d/code_md5.txt" 2>/dev/null || true
python3 "$HOME/EV1_score.py" "$d" --json "$d/ev1_score.json" > "$d/ev1_score.log" 2>&1 || true

# End only experiment processes that appeared after this run began. PID 1, 30,
# 36 and this helper's own shell are always excluded. No broad name-based kill.
current="$(ps -eo pid=,args=)"
while read -r pid args; do
  [ -n "${pid:-}" ] || continue
  case " $before " in *" $pid "*) continue ;; esac
  [ "$pid" = "1" ] || [ "$pid" = "30" ] || [ "$pid" = "36" ] || [ "$pid" = "$$" ] || [ "$pid" = "$PPID" ] || {
    case "$args" in
      *race_stack/install*|*ekf_node*|*gym_bridge*|*particle_filter*|*"ros2 launch"*|*map_server*|*lifecycle_manager*|*robot_state_publisher*|*EV2_sim_eval.py*|*gym_bridge*) kill -9 "$pid" 2>/dev/null || true ;;
    esac
  }
done < <(printf '%s\n' "$current" | awk 'NF {pid=$1; $1=""; sub(/^ /,""); print pid, $0}')

end="$(date +%s%N)"
elapsed="$(python3 -c 'import sys; print((int(sys.argv[2])-int(sys.argv[1]))/1e9)' "$start" "$end")"
python3 - "$d" "$base" "$seed" "$obs" "$host_load" "$elapsed" "$run_rc" <<'PY'
import csv, glob, json, math, os, statistics, sys
d, base, seed, obs, host, elapsed, rc = sys.argv[1:]
score_path=os.path.join(d,'ev1_score.json')
try: score=json.load(open(score_path,encoding='utf-8'))
except Exception: score={}
values=[]
try:
    with open(os.path.join(d,'trace.csv'),newline='',encoding='utf-8') as f:
        for row in csv.DictReader(f):
            try: values.append(float(row['step_ms']))
            except (ValueError,KeyError): pass
except OSError: pass
values.sort()
ex=sum(x>25 for x in values)
trace={'n':len(values),'mean_ms':statistics.fmean(values) if values else None,
       'p95_ms':values[min(len(values)-1,math.ceil(.95*len(values))-1)] if values else None,
       'gt25_n':ex,'gt25_pct':100*ex/len(values) if values else None}
collision0=False
for path in glob.glob(os.path.join(d,'race_result_*.json')):
    try: rr=json.load(open(path,encoding='utf-8'))
    except Exception: continue
    for ev in rr.get('collision_events',[]):
        t=float(ev.get('t',999))
        x=float(ev.get('x',999)); y=float(ev.get('y',999))
        if t <= 1.0 and math.hypot(x-.47,y-.30) <= .15: collision0=True
reason=score.get('reason') or ('setup_error' if not values else 'unknown')
res={'name':base,'seed':int(seed),'obstacles':int(obs),'scale':0.7,'package':'/tmp/pkg_h4_stuck',
     'host_cpu_load_start':host,'elapsed_wall_s':float(elapsed),'runner_exit_code':int(rc),
     'reason':reason,'clean_laps':score.get('clean_laps'),'laps_completed':score.get('laps_completed'),
     'best_lap_clean':score.get('best_lap_clean'),'collision_events':score.get('collision_events'),
     'collision_at_t0':collision0,'trace':trace,
     'stalls_free':score.get('stalls_free',[]),'err_pf_vs_gt_m':score.get('err_pf_vs_gt_m')}
json.dump(res,open(os.path.join(d,'ST_metrics.json'),'w',encoding='utf-8'),ensure_ascii=False,indent=2)
print('RESULT='+json.dumps(res,ensure_ascii=False,separators=(',',':')))
PY
exit 0
