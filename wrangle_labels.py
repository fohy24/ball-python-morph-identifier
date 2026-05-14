"""Wrangle all CSVs in data/labels/ into the format of data/selected_gene_df.csv.

Columns H onwards are one-hot binary columns, one per unique gene found across
the union of the `genes` lists (heterozygous genes excluded).
"""
import argparse
import ast
import os

import numpy as np
import pandas as pd


def load_labels(folder_path: str) -> pd.DataFrame:
    frames = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith(".csv"):
            frames.append(pd.read_csv(os.path.join(folder_path, filename)))
    return pd.concat(frames, ignore_index=True)


def clean_genes(labels: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    labels.loc[labels["genes"] == "[]", "genes"] = '["Normal"]'

    parsed = [ast.literal_eval(g) for g in labels["genes"]]
    no_het = [[item for item in lst if "het" not in item.lower()] for lst in parsed]
    no_het = [["Normal"] if not lst else lst for lst in no_het]

    labels = labels.copy()
    labels["genes"] = no_het

    union = sorted({gene for lst in no_het for gene in lst})
    return labels, union


def expand_genes(labels: pd.DataFrame, gene_columns: list[str]) -> pd.DataFrame:
    extension = pd.DataFrame(
        np.zeros([labels.shape[0], len(gene_columns)], dtype=int),
        columns=gene_columns,
    )
    extended = pd.concat([labels.reset_index(drop=True), extension], axis=1)

    for row_idx, gene_list in enumerate(labels["genes"]):
        for gene in gene_list:
            extended.at[row_idx, gene] = 1

    assert list(extended[gene_columns].sum(axis=1)) == [len(lst) for lst in labels["genes"]]
    return extended


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="data/labels/")
    parser.add_argument("--output", default="data/wrangled_gene_df.csv")
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Keep only the K most common morphs as columns; drop listings with any morph outside that set.",
    )
    args = parser.parse_args()

    labels = load_labels(args.input_dir)
    print(f"Loaded {labels.shape[0]} rows from {args.input_dir}")

    labels, gene_columns = clean_genes(labels)
    print(f"Found {len(gene_columns)} unique genes (het excluded)")

    if args.top_k is not None:
        gene_counts = pd.Series(
            [g for lst in labels["genes"] for g in lst]
        ).value_counts()
        top_k = list(gene_counts.head(args.top_k).index)
        top_k_set = set(top_k)
        mask = labels["genes"].apply(lambda lst: set(lst).issubset(top_k_set))
        labels = labels.loc[mask].reset_index(drop=True)
        gene_columns = top_k
        print(f"Filtered to top-{args.top_k} morphs: {labels.shape[0]} rows kept")

    extended = expand_genes(labels, gene_columns)
    extended.to_csv(args.output, index=False)
    print(f"Wrote {extended.shape} to {args.output}")


if __name__ == "__main__":
    main()
