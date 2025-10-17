
import polars as pl
import numpy as np
from pathlib import Path
from torchvision import datasets

DATA_DIR = Path('data')
ARTIFACTS = Path('artifacts')
DATA_DIR.mkdir(exist_ok=True, parents=True)
ARTIFACTS.mkdir(exist_ok=True, parents=True)

def compute_features(images: np.ndarray) -> pl.DataFrame:
    flat = images.reshape(len(images), -1).astype(np.float32) / 255.0
    mean = flat.mean(axis=1)
    var = flat.var(axis=1)
    # 16-bin histogram per image
    bins = np.linspace(0, 1, 17)
    hist = np.apply_along_axis(lambda r: np.histogram(r, bins=bins, density=True)[0], 1, flat)
    df = pl.DataFrame({
        'image_id': np.arange(len(images), dtype=np.int64),
        'pix_mean': mean,
        'pix_var': var,
    })
    for i in range(16):
        df = df.with_columns(pl.Series(f'hist_{i}', hist[:, i]))
    return df

def validate(df: pl.DataFrame):
    assert df.select(pl.col('pix_mean').is_between(0,1).all()).item(), 'pix_mean out of [0,1]'
    assert df.select(pl.col('pix_var').is_between(0,1).all()).item(), 'pix_var out of [0,1]'
    # simple sanity: histogram rows sum to ~1 (density)
    hist_cols = [f'hist_{i}' for i in range(16)]
    sums = df.select(pl.sum_horizontal(hist_cols).alias('s')).to_series()
    assert ((sums > 0.9) & (sums < 1.1)).all(), 'hist density not normalized'

def main():
    ds = datasets.MNIST(root=str(DATA_DIR), train=True, download=True)
    images = ds.data.numpy()  # (N, 28, 28) uint8
    labels = ds.targets.numpy().astype(np.int64)
    feats = compute_features(images)
    feats = feats.with_columns(pl.Series('label', labels))
    validate(feats)
    out_path = ARTIFACTS / 'features.parquet'
    feats.write_parquet(out_path)
    print(f'Wrote features to {out_path.resolve()}')

if __name__ == '__main__':
    main()
