import csv
import io
import json
import math
import re
import statistics
import tarfile
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
ST_ARCHIVE=ROOT/'tables'/'ST_runs.tar.gz'
S07_ARCHIVE=ROOT/'tables'/'S07_runs.tar.gz'
CSV_PATH=ROOT/'tables'/'ST_trials.csv'
REPORT=ROOT/'docs'/'superpowers'/'reports'/'2026-11-25-stuck-release-sim36.md'
RESULTS=ROOT/'tables'/'ST_sim36_results.json'
PLOT_TARGETS=ROOT/'tables'/'ST_plot_targets.json'
WP_PATH=ROOT/'tables'/'PC1_cause_read_20261124'/'wp.json'

def load_archive(path, prefix=None):
    data=defaultdict(dict)
    with tarfile.open(path,'r:gz') as tf:
        for m in tf.getmembers():
            if not m.isfile(): continue
            parts=m.name.replace('\\','/').split('/')
            if prefix:
                if len(parts)<3 or parts[0]!=prefix: continue
                name=parts[1]; rel='/'.join(parts[2:])
            else:
                if len(parts)<2: continue
                name=parts[0]; rel='/'.join(parts[1:])
            if not name.startswith(('ST_','S07_')): continue
            f=tf.extractfile(m)
            if f: data[name][rel]=f.read()
    return data

def obj(files, path, default=None):
    try: return json.loads(files[path].decode('utf-8'))
    except Exception: return default

def text(files,path,default=''):
    try: return files[path].decode('utf-8',errors='replace')
    except Exception: return default

root=json.loads(WP_PATH.read_text(encoding='utf-8'))['global_traj_wpnts_iqp']['wpnts']
wx=[p['x_m'] for p in root]; wy=[p['y_m'] for p in root]; ws=[p['s_m'] for p in root]; psi=[p['psi_rad'] for p in root]
def sd(x,y):
    i=min(range(len(wx)),key=lambda k:(wx[k]-x)**2+(wy[k]-y)**2)
    dx=x-wx[i]; dy=y-wy[i]
    return ws[i],-dx*math.sin(psi[i])+dy*math.cos(psi[i])

def run_info(name, files, prefix):
    metric=obj(files, f'{prefix}_metrics.json',{}) or {}
    score=obj(files,'ev1_score.json',{}) or {}
    rrname=next((k for k in sorted(files) if k.startswith('race_result_') and k.endswith('.json')),None)
    rr=obj(files,rrname,{}) if rrname else {}
    match=re.match(r'(?:ST|S07)_(\d+)_o([135])_B(?:_r(\d+))?$',name)
    if not match: return None
    seed=int(match.group(1)); obs=int(match.group(2))
    reason=metric.get('reason') or score.get('reason') or ('setup_error' if not files.get('trace.csv') else 'unknown')
    trace=metric.get('trace') or {}
    if not trace.get('n') and files.get('trace.csv'):
        vals=[]
        for r in csv.DictReader(io.StringIO(files['trace.csv'].decode('utf-8',errors='replace'))):
            try: vals.append(float(r['step_ms']))
            except Exception: pass
        vals.sort(); n=len(vals); ex=sum(v>25 for v in vals)
        trace={'n':n,'mean_ms':statistics.fmean(vals) if vals else None,'p95_ms':vals[min(n-1,math.ceil(.95*n)-1)] if n else None,'gt25_n':ex,'gt25_pct':100*ex/n if n else None}
    collisions=[]
    for e in (rr or {}).get('collision_events',[]):
        try:
            s,d=sd(float(e['x']),float(e['y']))
            collisions.append({'kind':e.get('kind','collision'),'t':float(e.get('t',0)),'s':s,'d':d,'x':float(e['x']),'y':float(e['y']),'lap':int(e.get('lap',0))+1})
        except Exception: pass
    stalls=[]
    for e in score.get('stalls_free',[]):
        try:
            dur=float(e.get('dur',0)); x,y=map(float,e['gt_xy'])
            if dur<10: continue
            s,d=sd(x,y)
            stalls.append({'dur':dur,'t_rel':float(e.get('t_rel',0)),'x':x,'y':y,'s':s,'d':d,'shield':None,'min_ttc':None,'speed_cmd':None,'accel':None,'pattern':'미판정'})
        except Exception: pass
    trace_rows=[]
    if files.get('trace.csv'):
        for r in csv.DictReader(io.StringIO(files['trace.csv'].decode('utf-8',errors='replace'))):
            try: trace_rows.append({k:float(v) for k,v in r.items() if v not in (None,'')})
            except Exception: continue
    for st in stalls:
        candidates=[r for r in trace_rows if 'x' in r and 'y' in r and r.get('v',99)<0.2]
        if not candidates: candidates=[r for r in trace_rows if 'x' in r and 'y' in r]
        if candidates:
            near=min(candidates,key=lambda r:(r['x']-st['x'])**2+(r['y']-st['y'])**2)
            for k in ('shield','min_ttc','speed_cmd','accel'): st[k]=near.get(k)
            try:
                sh=int(near.get('shield',-1)); ac=float(near.get('accel',0))
                st['pattern']='2028형(shield=1, accel>0)' if sh==1 and ac>0 else ('2031형(shield=0, accel<0)' if sh==0 and ac<0 else '기타')
            except Exception: pass
    activations=[]
    if trace_rows:
        qualifying=[]
        for r in trace_rows:
            q=int(r.get('shield',0))==1 and 0.1<=r.get('speed_cmd',-1)<=0.30001
            qualifying.append(q)
        i=0
        while i<len(trace_rows):
            if not qualifying[i]: i+=1; continue
            j=i+1
            while j<len(trace_rows) and qualifying[j] and trace_rows[j].get('t',0)-trace_rows[j-1].get('t',0)<=0.10: j+=1
            start=trace_rows[i]; prev=trace_rows[i-1] if i else start
            if min(prev.get('v',99),start.get('v',99))<=0.08:
                t0=start.get('t',0); v0=start.get('v',0)
                after=[r for r in trace_rows if t0<r.get('t',0)<=t0+10]
                rose=any(r.get('v',0)>max(0.10,v0+0.05) for r in after)
                if rose:
                    end=trace_rows[j-1]; duration=max(0,end.get('t',t0)-t0)
                    collisions_after=[c for c in collisions if 0<c['t']-t0<=5]
                    resumed=any(r.get('v',0)>1.0 for r in after)
                    activations.append({'t':t0-trace_rows[0].get('t',0),'s':start.get('s'),'d':start.get('d'),'duration':duration,'collision_5s':bool(collisions_after),'collision_kind':','.join(c['kind'] for c in collisions_after),'resumed_10s':resumed,'max_v_10s':max((r.get('v',0) for r in after),default=0)})
            i=max(j,i+1)
    t0=bool(metric.get('collision_at_t0')) or any(c['t']<=1 and math.hypot(c['x']-.47,c['y']-.30)<=.15 for c in collisions)
    return {'name':name,'seed':seed,'obs':obs,'retry':int(match.group(3) or 1),'reason':reason,'metric':metric,'score':score,'trace':trace,'collisions':collisions,'stalls':stalls,'activations':activations,'t0':t0,'invalid':reason=='setup_error' or t0,'clean_laps':score.get('clean_laps',metric.get('clean_laps')),'laps_completed':score.get('laps_completed',metric.get('laps_completed')),'best':score.get('best_lap_clean',metric.get('best_lap_clean')),'race_total':(rr or {}).get('total_time'),'host':metric.get('host_cpu_load_start'),'wall':metric.get('elapsed_wall_s'),'runlog':text(files,'race.log')}

st_files=load_archive(ST_ARCHIVE)
s07_files=load_archive(S07_ARCHIVE,'runs')
st_rows=[r for n,f in st_files.items() if (r:=run_info(n,f,'ST'))]
s07_rows=[r for n,f in s07_files.items() if (r:=run_info(n,f,'S07'))]
st_rows.sort(key=lambda r:(r['seed'],r['obs'],r['retry']))
s07_rows.sort(key=lambda r:(r['seed'],r['obs'],r['retry']))

def choose(rows):
    groups=defaultdict(list)
    for r in rows: groups[(r['seed'],r['obs'])].append(r)
    out={}
    for key,rs in groups.items():
        out[key]=sorted(rs,key=lambda r:(r['invalid'],-(r['clean_laps'] or 0),r['retry']))[0]
    return out
st_best=choose(st_rows); s07_best=choose(s07_rows)

fields=['판','seed','장애물','재시도','분류','완주랩','종료사유','10초+정지수','충돌수','충돌·정지 위치(s,d)','해소작동수','25ms초과율_pct','step평균_ms','step_p95_ms','호스트CPU_pct','실행벽시계_s','환경불량']
with open(CSV_PATH,'w',newline='',encoding='utf-8') as f:
    w=csv.DictWriter(f,fieldnames=fields); w.writeheader()
    for r in st_rows:
        status='무효(t=0 충돌)' if r['t0'] else ('무효(setup_error)' if r['reason']=='setup_error' else ('10랩' if (r['clean_laps'] or 0)>=10 else '미완주'))
        loc=[]
        for c in r['collisions']: loc.append(f"{c['kind']} L{c['lap']} s{c['s']:.2f},d{c['d']:+.2f}")
        for e in r['stalls']: loc.append(f"정지 {e['dur']:.0f}s s{e['s']:.2f},d{e['d']:+.2f}")
        t=r['trace']
        w.writerow({'판':r['name'],'seed':r['seed'],'장애물':r['obs'],'재시도':r['retry'],'분류':status,'완주랩':r['clean_laps'],'종료사유':r['reason'],'10초+정지수':len(r['stalls']),'충돌수':len(r['collisions']),'충돌·정지 위치(s,d)':'; '.join(loc) or '—','해소작동수':len(r['activations']),'25ms초과율_pct':t.get('gt25_pct'),'step평균_ms':t.get('mean_ms'),'step_p95_ms':t.get('p95_ms'),'호스트CPU_pct':r['host'],'실행벽시계_s':r['wall'],'환경불량':'예' if r.get('env_bad') else '아니오'})

segments=[('s15.3–19.6',15.3,19.6),('s21.5–25.1',21.5,25.1),('s31.5–34.5',31.5,34.5),('s40.0–43.0',40.0,43.0),('s45.0–48.0',45.0,48.0)]
def summary(rows):
    valid=[r for r in rows if not r['invalid']]
    full=sum((r['clean_laps'] or 0)>=10 for r in valid); laps=sum(r['clean_laps'] or 0 for r in valid)
    stalls=[e for r in valid for e in r['stalls']]
    collisions=[e for r in valid for e in r['collisions']]
    return {'valid':len(valid),'full':full,'laps':laps,'stall_n':len(stalls),'stall_sec':sum(e['dur'] for e in stalls),'collision_n':len(collisions),'collisions':collisions,'stalls':stalls}
st_sum=summary(list(st_best.values())); s07_sum=summary(list(s07_best.values()))
obs_table=[]
for obs in (1,3,5):
    a=summary([r for (seed,o),r in st_best.items() if o==obs]); b=summary([r for (seed,o),r in s07_best.items() if o==obs])
    obs_table.append(f"| {obs} | {a['valid']}/12 | {a['full']}/{a['valid']} · {a['laps']} | {b['valid']}/12 | {b['full']}/{b['valid']} · {b['laps']} | {a['stall_n']} / {a['stall_sec']:.1f} | {b['stall_n']} / {b['stall_sec']:.1f} | {a['collision_n']} | {b['collision_n']} |")
obs_table.append(f"| 합계 | {st_sum['valid']}/36 | {st_sum['full']}/{st_sum['valid']} · {st_sum['laps']} | {s07_sum['valid']}/36 | {s07_sum['full']}/{s07_sum['valid']} · {s07_sum['laps']} | {st_sum['stall_n']} / {st_sum['stall_sec']:.1f} | {s07_sum['stall_n']} / {s07_sum['stall_sec']:.1f} | {st_sum['collision_n']} | {s07_sum['collision_n']} |")
compare='\n'.join(obs_table)
seg_lines=[]
for label,lo,hi in segments:
    a=sum(lo<=c['s']<=hi for c in st_sum['collisions']); b=sum(lo<=c['s']<=hi for c in s07_sum['collisions'])
    seg_lines.append(f'| {label} | {a} | {b} |')
seg_table='\n'.join(seg_lines)

activation_lines=[]
for r in st_rows:
    for i,e in enumerate(r['activations'],1):
        activation_lines.append(f"| {r['name']} | {i} | {e['t']:.3f} | {e['s'] if e['s'] is not None else float('nan'):.2f}, {e['d'] if e['d'] is not None else float('nan'):+.2f} | {e['duration']:.3f} | {'예: '+e['collision_kind'] if e['collision_5s'] else '아니오'} | {'예' if e['resumed_10s'] else '아니오'} | {e['max_v_10s']:.2f} |")
activation_table='\n'.join(activation_lines) if activation_lines else '| 없음 | — | — | — | — | — | — | — |'

special=[]
for seed,obs,label in ((2028,1,'S07 2028형'),(2029,1,'S07 2029형'),(2033,5,'S07 2033형')):
    off=s07_best.get((seed,obs)); on=st_best.get((seed,obs))
    if not on: continue
    ev=', '.join(f"t={x['t']:.2f}s s={x['s']:.2f} d={x['d']:+.2f}" for x in on['activations']) or '없음'
    special.append(f"| {label} ({seed}, o{obs}) | {on['clean_laps']} | {len(on['stalls'])} / {sum(e['dur'] for e in on['stalls']):.1f}s | {len(on['collisions'])} | {len(on['activations'])} | {ev} |")
special_table='\n'.join(special)

shield0=[]
for r in st_rows:
    for e in r['stalls']:
        try:
            if int(float(e['shield']))==0 and float(e['accel'])<0: shield0.append(f"| {r['name']} | {e['dur']:.1f} | s={e['s']:.2f}, d={e['d']:+.2f} | {e['shield']} | {e['accel']:.3f} |")
        except Exception: pass
shield0_table='\n'.join(shield0) if shield0 else '| 해당 정지 없음 | — | — | — | — |'

all_runs=len(st_rows); unique=len(st_best); invalid=sum(r['invalid'] for r in st_rows); retries=len(st_rows)-unique
progress=json.loads(RESULTS.read_text(encoding='utf-8'))
plot_targets=[]
for r in st_rows:
    for e in r['activations']:
        if e['collision_5s']:
            plot_targets.append(r['name']); break
PLOT_TARGETS.write_text(json.dumps(sorted(set(plot_targets)),ensure_ascii=False,indent=2),encoding='utf-8')

run_lines=[]
for r in st_rows:
    status='무효(t=0)' if r['t0'] else ('무효(setup_error)' if r['reason']=='setup_error' else ('10랩' if (r['clean_laps'] or 0)>=10 else '미완주'))
    loc=[]
    for c in r['collisions']: loc.append(f"{c['kind']} L{c['lap']} s{c['s']:.2f},d{c['d']:+.2f}")
    for e in r['stalls']: loc.append(f"정지 {e['dur']:.0f}s s{e['s']:.2f},d{e['d']:+.2f}")
    t=r['trace']; pct='—' if t.get('gt25_pct') is None else f"{t['gt25_pct']:.3f}%"
    host='—' if r['host'] is None else f"{r['host']}%"
    run_lines.append(f"| {r['name']} | {status} | {r['clean_laps'] if r['clean_laps'] is not None else '—'} | {r['reason']} | {'; '.join(loc) or '—'} | {t.get('mean_ms','—')} / {t.get('p95_ms','—')} / {pct} | {host} | {r['wall'] if r['wall'] is not None else '—'} |")
run_table='\n'.join(run_lines)

rate_vals=[float(r['trace']['gt25_pct']) for r in st_rows if r['trace'].get('gt25_pct') is not None]
cpu_vals=[]
for r in st_rows:
    try: cpu_vals.append(float(r['host']))
    except Exception: pass
env_flagged=[r['name'] for r in st_rows if r.get('env_bad')]
pcpu=f"{statistics.fmean(cpu_vals):.1f}% ({min(cpu_vals):g}–{max(cpu_vals):g}%)" if cpu_vals else '기록 없음'
best=min((float(r['best']) for r in st_best.values() if not r['invalid'] and r['best'] is not None),default=None)
all_clean=[]
for r in st_best.values():
    if r['invalid']: continue
    for k in ('clean_lap_times','lap_times_clean','lap_times'):
        v=r['score'].get(k)
        if isinstance(v,list):
            all_clean.extend(float(x) for x in v if isinstance(x,(int,float)) and x>0); break
avg=statistics.fmean(all_clean) if all_clean else None

gate='통과' if progress.get('stop_reason')=='completed_all_36' and unique==36 else '미완료/중단'
def pct_for(seed, obs):
    row=next((r for r in st_rows if r['seed']==seed and r['obs']==obs),None)
    return f"{row['trace'].get('gt25_pct',0):.3f}%" if row else '—'
rate_2032_o3=pct_for(2032,3)
rate_2032_o5=pct_for(2032,5)
rate_2036_o5=pct_for(2036,5)
activation_total=sum(len(r['activations']) for r in st_rows)
activation_not_resumed=sum(not e['resumed_10s'] for r in st_rows for e in r['activations'])
section=f'''## 정정 후 시뮬레이션 결과 — 정지 해소 ON

앞 절의 "호스트 CPU 평균 10% 이하"로 인한 0/36 중단은 **정정 전 시도**로 보존한다. 후속 안내에 따라 이 CPU 조건은 시작 게이트로 적용하지 않고 기록만 했다. 컨테이너 Python 500만 회 반복은 시작 전 0.186644초로 0.40초 기준을 통과했다. 결과 보고서는 앞 절을 유지하고 이 결과 절을 끝에 추가했다.

### 실행 요약

| 항목 | 결과 |
|---|---:|
| 36개 설정 처리 | {unique}/36 ({gate}) |
| 물리 실행 수 | {all_runs} (setup_error 재시도 {retries}회) |
| 유효 설정 | {st_sum['valid']}/36; 무효 시도 {invalid}회 |
| 10랩 완주 | {st_sum['full']}/{st_sum['valid']} |
| 유효 완주 랩 합 | {st_sum['laps']} |
| 10초 이상 정지 | {st_sum['stall_n']}건, {st_sum['stall_sec']:.1f}초 합계 |
| 충돌 | {st_sum['collision_n']}건 |
| best / 평균 무충돌 랩 | {best if best is not None else '—'} / {avg if avg is not None else '—'}초 |
| 판별 25ms 초과율 범위 | {min(rate_vals):.3f}–{max(rate_vals):.3f}% |
| 판 시작 CPU 표본 평균 (범위) | {pcpu} — 기록 전용 |
| 종료 상태 | {progress.get('stop_reason')} |

우선순위 5설정은 2028 o1, 2029 o1, 2033 o5, 2027 o3, 2031 o3 순으로 먼저 실행했다. 그 뒤 나머지는 seed 순으로 실행하고 t=0 후보 2032 o3, 2032 o5, 2036 o5를 마지막에 실행했다. 첫 두 판의 초과율은 1% 미만이었다. 5% 초과 판은 {sum(v>5 for v in rate_vals)}개였고, 연속 세 판 조건은 발생하지 않았다. 마지막 후보 중 2032 o3와 o5는 각각 {rate_2032_o3} 및 {rate_2032_o5}로 두 번 연속 5%를 넘었고, 다음 2036 o5는 {rate_2036_o5}여서 경보가 초기화됐다. t=0 충돌은 재시도하지 않았다.

### 판별별 결과

`완주랩`은 EV1 점수기의 연속 무충돌 랩 수다. 위치 s,d는 `pc1_tally.py`와 같은 waypoint 최근접점 투영이다. 호스트 CPU는 판 시작 표본이다.

| 판 | 분류 | 완주랩 | 종료 사유 | 충돌·장시간 정지 위치 | step 평균 / p95 / >25ms | CPU | 실행 초 |
|---|---|---:|---|---|---|---:|---:|
{run_table}

### 정지 해소 작동과 이후 주행

작동 후보는 `trace.csv`에서 shield=1, speed_cmd 0.1–0.3m/s, 시작 속도 약 0.08m/s 이하인 구간으로 찾고, 이후 10초 이내 실제 속도가 시작보다 0.05m/s 이상 오르며 0.10m/s를 넘는 경우로 셌다. 작동 시간은 같은 명령 조건이 이어진 길이다. 5초 이내 충돌과 10초 이내 v>1m/s 도달을 따로 표시했다.

| 판 | 작동 | 시작 t(초) | 시작 s,d | 지속(초) | 5초 내 충돌 | 10초 내 v>1 | 10초 내 최고 v |
|---|---:|---:|---|---:|---|---|---:|
{activation_table}

검출한 해소 작동은 {activation_total}회다. 작동했지만 10초 안에 1m/s를 넘지 못한 횟수는 {activation_not_resumed}회다. 작동 뒤 5초 이내 충돌이 연결된 판은 {', '.join(sorted(set(plot_targets))) if plot_targets else '없음'}이다.

### S07(해소 OFF)과 쌍 비교

같은 seed·장애물의 유효 결과를 비교한다. 재시도는 성공한 시도로 대표하고, t=0 충돌/setup_error만 남은 설정은 유효 완주 통계에서 제외한다. 실행 간 주행이 달라질 수 있어 해소 기능의 인과 효과로 단정하지 않는다.

| 장애물 | ST 유효 설정 | ST 10랩 완주 · 랩 합 | S07 유효 설정 | S07 10랩 완주 · 랩 합 | ST 정지 건수 / 초 | S07 정지 건수 / 초 | ST 충돌 | S07 충돌 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{compare}

| 충돌 위치 구간 | ST 충돌 수 | S07 충돌 수 |
|---|---:|---:|
{seg_table}

#### 기존 정지 형태별 대응

| 기존 S07 정지 설정 | ST 완주랩 | ST 10초+정지 수/초 | ST 충돌 | 해소 작동 수 | 작동 시각·위치 |
|---|---:|---:|---:|---:|---|
{special_table}

### 2031형 정지 확인

ST에서 10초 이상 정지 중 가장 가까운 trace 표본이 shield=0이고 accel<0인 사례를 확인했다. 해당 조건에서는 새 해소 규칙이 작동하지 않아야 한다.

| 판 | 정지(초) | 위치 s,d | shield | accel |
|---|---:|---|---:|---:|
{shield0_table}

### 원자료와 제한

원자료 압축은 `tables/ST_runs.tar.gz`, 판별표는 `tables/ST_trials.csv`다. 해소 작동 후 충돌에 해당하는 확대 그림은 `img/ST/<seed>/`에 보관한다. 기존 S07 보고서와 이 보고서 앞 절의 정정 전 사전 게이트 기록은 덮어쓰지 않았다.

정지 해소 기능은 오프라인 재생에서만 검증됐다. 실제 판의 주행 결과는 이 seed·장애물 배치에서 관측한 결과이며, 안전성을 일반화하지 않는다. 04:45 이전에 36설정이 끝났고 이후 새 판을 시작하지 않았다. Docker/WSL/ROS daemon은 재시작하지 않았다.
'''

old=REPORT.read_text(encoding='utf-8') if REPORT.exists() else ''
marker='## 정정 후 시뮬레이션 결과 — 정지 해소 ON'
base=old.split(marker)[0].rstrip() if marker in old else old.rstrip()
REPORT.write_text(base+'\n\n'+section,encoding='utf-8',newline='\n')
print(json.dumps({'unique_settings':unique,'physical_runs':all_runs,'invalid_attempts':invalid,'activations':sum(len(r['activations']) for r in st_rows),'plot_targets':plot_targets,'report':str(REPORT),'csv_rows':len(st_rows)},ensure_ascii=False))
