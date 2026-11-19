"""Offline timing only: does not launch ROS or claim closed-loop success.

python3 -m mppi_racer.benchmark --map PATH --output timings.json
"""
import argparse
import json
import os
import platform
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .mppi import MPPI, MPPIParams
from .track import Track


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--map', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--scale', type=float, default=0.70)
    parser.add_argument('--steps', type=int, default=100)
    parser.add_argument('--samples', type=int, default=1024)
    args = parser.parse_args()
    if not 0 < args.scale <= 1 or args.steps < 1 or args.samples < 1:
        parser.error('scale must be in (0,1]; steps and samples must be positive')
    data = json.loads(Path(args.map).read_text())
    w = data['global_traj_wpnts_iqp']['wpnts']
    if np.hypot(w[0]['x_m']-w[-1]['x_m'], w[0]['y_m']-w[-1]['y_m']) < 1e-3:
        w = w[:-1]
    def column(name):
        return np.array([p[name] for p in w])
    tr = Track(column('x_m'), column('y_m'), column('vx_mps')*args.scale,
               column('d_left'), column('d_right'))
    params = MPPIParams(samples=args.samples)
    report = dict(kind='offline fixed-state workload; not driving validation',
                  python=platform.python_version(), numpy=np.__version__,
                  cpu=platform.machine(), scale=args.scale, params=asdict(params),
                  threads={k: os.environ.get(k) for k in
                           ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS')},
                  scenarios={})
    for count in (0, 4):
        ctrl = MPPI(tr, params, seed=2026)
        # Synthetic perceived rectangles, deliberately separate from simulator GT.
        obs = np.array([[s, s+.3, -.15, .15, 0, 0] for s in (3.,6.,9.,12.)][:count]).reshape(-1,6)
        rows = []
        for i in range(args.steps+10):
            j = (i*7) % tr.n
            state = [tr.x[j], tr.y[j], tr.psi[j], tr.vx[j]]
            timings = {}
            ctrl.step(state, obs, profile=timings)
            if i >= 10:
                rows.append(timings)
        summary = {}
        for key in rows[0]:
            values = np.array([r[key] for r in rows])
            summary[key] = dict(mean=float(values.mean()), p95=float(np.percentile(values,95)),
                                p99=float(np.percentile(values,99)), max=float(values.max()))
        report['scenarios'][str(count)] = dict(summary=summary, rows=rows)
    Path(args.output).write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps({k: v['summary'] for k,v in report['scenarios'].items()}, indent=2))


if __name__ == '__main__':
    main()
