#!/usr/bin/env python3
"""Color-coded log-log FLOP/s vs n for all 13 physics kernels -> sizemap.pdf.

Usage: ~/venv/bin/python3 plot.py    (reads *.dat next to this file)
"""
import glob
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

here = os.path.dirname(os.path.abspath(__file__))
series = []
for f in sorted(glob.glob(os.path.join(here, '*.dat'))):
    op = os.path.basename(f)[:-4]
    pts = [(int(n), float(v)) for n, v in
           (l.split() for l in open(f) if not l.startswith('#'))]
    ns, fl = (list(t) for t in zip(*pts))
    series.append((op, (ns, fl)))
series.sort(key=lambda s: -max(s[1][1]))

fig, ax = plt.subplots(figsize=(9, 5.5))
for op, (ns, fl) in series:
    ax.plot(ns, fl, marker='o', ms=4, lw=1.6, label=f'{op} '
            f'({max(fl):.0e})')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('elements per vector, n')
ax.set_ylabel('FLOP/s')
ax.set_title('OpenBLAS physics kernels: throughput vs problem size\n'
             'double precision, 1 core — L2 peak at n≈10²–10³, '
             'DRAM plateau from n≈10⁵')
ax.grid(True, which='major', alpha=0.4)
ax.grid(True, which='minor', alpha=0.15)
ax.legend(fontsize=7.5, ncol=2, loc='lower left', framealpha=0.9)
fig.tight_layout()
out = os.path.join(here, 'sizemap.pdf')
fig.savefig(out)
print('wrote', out)
