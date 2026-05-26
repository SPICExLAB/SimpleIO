"""
Summarize per-sequence eval results from evaluate.py.
"""

import argparse
import json
import statistics as s


def hr(width=110):
    print('─' * width)


def summarize(path, table=False):
    results = json.load(open(path))
    keys = ['ATE', 'ATE_H', 'ATE_V', 'RP_RMSE', 'drift_rate']

    print(f"\n{path}  ({len(results)} sequences)")
    hr()
    print(f"{'metric':>10s}  {'mean':>8s} {'median':>8s} {'max':>8s}  worst")
    hr()
    for k in keys:
        vals = [r[k] for r in results]
        worst = max(results, key=lambda r: r[k])
        print(f"{k:>10s}  {s.mean(vals):8.3f} {s.median(vals):8.3f} {max(vals):8.3f}  {worst['name'][:50]}")
    hr()
    print(f"ATE > 10 m:      {sum(1 for r in results if r['ATE'] > 10):3d} / {len(results)}")
    print(f"drift_rate > 5%: {sum(1 for r in results if r['drift_rate'] > 5):3d} / {len(results)}")

    if table:
        print()
        hr(95)
        print(f"{'seq':50s} {'ATE':>8s} {'ATE_H':>8s} {'ATE_V':>8s} {'RTE':>8s} {'drift%':>8s}")
        hr(95)
        for r in sorted(results, key=lambda r: r['ATE'], reverse=True):
            print(f"{r['name'][:50]:50s} {r['ATE']:8.3f} {r['ATE_H']:8.3f} {r['ATE_V']:8.3f} {r['RP_RMSE']:8.3f} {r['drift_rate']:8.2f}")
        hr(95)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', type=str, default='result/loss_result/result.json')
    parser.add_argument('--table', action='store_true', help='print per-sequence table')
    args = parser.parse_args()

    summarize(args.json, table=args.table)
