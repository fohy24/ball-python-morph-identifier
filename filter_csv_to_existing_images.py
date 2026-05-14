"""Drop CSV rows whose referenced image isn't on disk.

Run on the training VM: `python3 /mnt/data/filter_csv_to_existing_images.py`
"""
import os
import sys

import pandas as pd

CSV = "/mnt/data/top37_gene_df.csv"
IMG_DIR = "/mnt/data/img"

df = pd.read_csv(CSV)
n_before = len(df)
print(f"Before: {n_before:,} rows")

exists = df.iloc[:, 0].astype(str).apply(
    lambda i: os.path.exists(os.path.join(IMG_DIR, f"{i}.png"))
)
n_missing = int((~exists).sum())
print(f"Missing image: {n_missing:,} rows ({n_missing/n_before*100:.3f}%)")

if n_missing == 0:
    print("Nothing to filter — exiting.")
    sys.exit(0)

df_clean = df[exists].reset_index(drop=True)
df_clean.to_csv(CSV, index=False)
print(f"After:  {len(df_clean):,} rows  -> {CSV}")
