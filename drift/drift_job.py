
import polars as pl
import numpy as np
from pathlib import Path

ARTIFACTS = Path('artifacts')
REF = ARTIFACTS / 'features.parquet'
INF = ARTIFACTS / 'inference_log.parquet'  # optional future logging

def kl_div(p, q, eps=1e-8):
    p = np.clip(p, eps, 1)
    q = np.clip(q, eps, 1)
    p = p / p.sum(); q = q / q.sum()
    return np.sum(p * np.log(p / q))

def main():
    if not REF.exists():
        print('No reference features yet. Run feature pipeline first.')
        return
    ref = pl.read_parquet(REF)
    ref_hist = ref.select([f'hist_{i}' for i in range(16)]).to_numpy().mean(axis=0)
    # In a real setup, collect recent inference distributions; here we reuse a subset
    cur = ref.sample(n=500, seed=42)
    cur_hist = cur.select([f'hist_{i}' for i in range(16)]).to_numpy().mean(axis=0)
    score = kl_div(ref_hist, cur_hist)
    print(f'Drift KL(ref||cur) = {score:.6f}')

if __name__ == '__main__':
    main()
