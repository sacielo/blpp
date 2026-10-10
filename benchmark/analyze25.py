#!/usr/bin/env python3
"""25-run size sweep plot: median FLOP/s with 4th-96th percentile bands.

Reads runs-<hostname>/rep_NN.tsv next to this file (columns: rep n op ms
flop_s, produced by sweep25.sh), writes stats.csv + sizemap25-<hostname>.
pdf/.png into bench25-<hostname>/ (gitignored). One set of folders per
system so results from different machines can coexist. Three side-by-side
panels with identical log-log axes; kernels grouped by shape of the
computation.

Usage: python3 analyze25.py     (needs numpy + matplotlib)
"""
import csv
import os
import socket
from collections import defaultdict

import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))
host = socket.gethostname().split('.')[0]
runs = os.path.join(here, f'runs-{host}')
out = os.path.join(here, f'bench25-{host}')
os.makedirs(out, exist_ok=True)

data = defaultdict(lambda: defaultdict(list))   # op -> n -> [flop_s]
samples = 0
for fn in sorted(os.listdir(runs)):
    if not fn.endswith('.tsv'):
        continue
    for line in open(os.path.join(runs, fn)):
        _, n, op, _ms, flop = line.split()
        data[op][int(n)].append(float(flop))
        samples += 1
if not samples:
    raise SystemExit(f'no data in {runs}; run sweep25.sh first')
print(f'{samples} samples, {len(data)} ops')

stats = []
agg = {}                           # (op, n) -> (med, lo, hi)
for op, pts in sorted(data.items()):
    for n, vals in sorted(pts.items()):
        med, lo, hi = np.percentile(vals, [50, 4, 96])
        agg[(op, n)] = (med, lo, hi)
        stats.append((op, n, med, lo, hi, len(vals)))
with open(os.path.join(out, 'stats.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['op', 'n', 'median', 'p4', 'p96', 'runs'])
    w.writerows(stats)

PANELS = [
    ('1-vector + pure dot/hadamard',
     ['daxpy', 'd1xypa', 'd1norm', 'd3dot', 'd3sqr', 'd3had']),
    ('cross-product kernels',
     ['d3cross', 'd3crossscal', 'd3crossdot', 'd3crosssqr',
      'd3crosscross', 'd3exb']),
    ('forks & fused chains',
     ['d3dotxy_dotxz', 'd3crossxy_crossxz', 'd3crossxy_dotxz',
      'd3norm_unit', 'd3refl', 'd3drag', 'd3mom_ke']),
]

fig, axes = plt.subplots(1, 3, figsize=(16, 5.4), sharex=True, sharey=True)
fig.suptitle('OpenBLAS physics kernels, 25 runs per point — median (line) '
             'with 4-96 percentile band (shaded); double, 1 core', y=1.0)
for ax, (title, ops) in zip(axes, PANELS):
    for op in ops:
        pts = sorted((n, v) for (o, n), v in agg.items() if o == op)
        if not pts:
            print('missing op:', op)
            continue
        ns = [n for n, _ in pts]
        med = [v[0] for _, v in pts]
        lo = [v[1] for _, v in pts]
        hi = [v[2] for _, v in pts]
        ax.plot(ns, med, marker='o', ms=3.5, lw=1.4, label=op)
        ax.fill_between(ns, lo, hi, alpha=0.15, lw=0)
    ax.set_title(title, fontsize=10)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel('elements per vector, n')
    ax.grid(True, which='major', alpha=0.4)
    ax.grid(True, which='minor', alpha=0.15)
    ax.legend(fontsize=7, loc='lower left', framealpha=0.9)
axes[0].set_ylabel('FLOP/s')
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(os.path.join(out, f'sizemap25-{host}.pdf'))
fig.savefig(os.path.join(out, f'sizemap25-{host}.png'), dpi=130)
print('wrote', out, f'/sizemap25-{host}.pdf/.png and stats.csv')
