"""Extrait un petit sous-ensemble de Sort of CLEVR pour la démo.

Crée data_demo_train.h5 / data_demo_val.h5 / data_demo_test.h5
dans le même dossier que ce script.

Usage :
    python sortofclevr/make_demo_subset.py \
        --src-dir C:/Users/ilies/Documents/cours/projFiLM/sortofclevr \
        --n-train 2000 --n-val 500 --n-test 500
"""

import argparse
from pathlib import Path

import h5py
import pandas as pd


def extract(src_h5: Path, src_csv: Path, dst_h5: Path, dst_csv: Path, n: int) -> None:
    df = pd.read_csv(src_csv).iloc[:n]
    df.to_csv(dst_csv, index=False)

    with h5py.File(src_h5, "r") as src:
        dataset_name = list(src.keys())[0]
        data = src[dataset_name][:n]

    with h5py.File(dst_h5, "w") as dst:
        dst.create_dataset(dataset_name, data=data, compression="gzip", compression_opts=6)

    print(f"  {dst_h5.name} ({dst_h5.stat().st_size // 1024} Ko) + {dst_csv.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-dir",  default="C:/Users/ilies/Documents/cours/projFiLM/sortofclevr")
    parser.add_argument("--dst-dir",  default=str(Path(__file__).parent / "demo"))
    parser.add_argument("--n-train",  type=int, default=2000)
    parser.add_argument("--n-val",    type=int, default=500)
    parser.add_argument("--n-test",   type=int, default=500)
    args = parser.parse_args()

    src = Path(args.src_dir)
    dst = Path(args.dst_dir)
    dst.mkdir(exist_ok=True)

    splits = [
        ("train", args.n_train),
        ("val",   args.n_val),
        ("test",  args.n_test),
    ]

    for split, n in splits:
        print(f"Extraction {split} ({n} samples)...")
        extract(
            src / f"data_{split}.h5",
            src / f"data_{split}.csv",
            dst / f"data_{split}.h5",
            dst / f"data_{split}.csv",
            n,
        )

    print(f"\nDone → {dst}/")


if __name__ == "__main__":
    main()
