"""Download a mini-subset of CLEVR v1.

Usage
-----
    python data/download_clevr.py                   # defaults
    python data/download_clevr.py --n-train 2000 --n-val 500

What it downloads
-----------------
1. CLEVR_v1.0_no_images.zip  (~18 MB) — questions JSON only
2. Individual PNG images from the CLEVR CDN (train + val)
   - Only the first --n-train / --n-val images are fetched.
   - Images are stored in data/clevr/images/{train,val}/

Output layout
-------------
data/clevr/
├── images/
│   ├── train/  CLEVR_train_XXXXXX.png
│   └── val/    CLEVR_val_XXXXXX.png
└── questions/
    ├── CLEVR_train_questions.json   (filtered to N first images)
    └── CLEVR_val_questions.json
"""

from __future__ import annotations

import argparse
import io
import json
import os
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm

# ---------------------------------------------------------------------------
# URLs
# ---------------------------------------------------------------------------
QUESTIONS_URL = "https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0_no_images.zip"
IMAGES_BASE_URL = "https://dl.fbaipublicfiles.com/clevr/CLEVR_v1.0/images"


def _download_file(url: str, dest: Path, desc: str = "") -> None:
    """Stream-download a file with a tqdm progress bar."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, stream=True, timeout=60)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))
    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=desc or dest.name
    ) as bar:
        for chunk in r.iter_content(chunk_size=1 << 16):
            f.write(chunk)
            bar.update(len(chunk))


def download_questions(data_dir: Path) -> None:
    """Download and extract the questions ZIP."""
    zip_path = data_dir / "CLEVR_v1.0_no_images.zip"
    questions_dir = data_dir / "questions"

    if questions_dir.exists() and any(questions_dir.glob("*.json")):
        print("[questions] Already present, skipping download.")
        return

    print("[questions] Downloading questions ZIP (~18 MB)...")
    _download_file(QUESTIONS_URL, zip_path, desc="questions.zip")

    print("[questions] Extracting...")
    with zipfile.ZipFile(zip_path) as zf:
        for member in zf.namelist():
            # Extract only the questions JSON files
            if "questions" in member and member.endswith(".json"):
                zf.extract(member, data_dir / "_tmp")
    # Move extracted files to questions_dir
    questions_dir.mkdir(parents=True, exist_ok=True)
    tmp_q = data_dir / "_tmp" / "CLEVR_v1.0" / "questions"
    if tmp_q.exists():
        for f in tmp_q.glob("*.json"):
            f.rename(questions_dir / f.name)
    # Cleanup
    zip_path.unlink(missing_ok=True)
    import shutil
    shutil.rmtree(data_dir / "_tmp", ignore_errors=True)
    print(f"[questions] Saved to {questions_dir}")


def filter_questions(questions_path: Path, n: int) -> list[dict]:
    """Load and return the first n questions whose images exist (or will exist)."""
    with open(questions_path) as f:
        data = json.load(f)
    questions = data["questions"]
    # Keep only questions for unique images, up to n images
    seen_images: set[str] = set()
    filtered = []
    for q in questions:
        img = q["image_filename"]
        seen_images.add(img)
        if len(seen_images) > n:
            break
        filtered.append(q)
    return filtered, sorted(seen_images)[:n]


def download_images(
    image_filenames: list[str],
    split: str,
    images_dir: Path,
) -> None:
    """Download individual images from the CLEVR CDN."""
    images_dir.mkdir(parents=True, exist_ok=True)
    print(f"[images/{split}] Downloading {len(image_filenames)} images...")
    for fname in tqdm(image_filenames, desc=f"images/{split}"):
        dest = images_dir / fname
        if dest.exists():
            continue
        url = f"{IMAGES_BASE_URL}/{split}/{fname}"
        try:
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            dest.write_bytes(r.content)
        except Exception as e:
            print(f"  Warning: could not download {fname}: {e}")


def save_filtered_questions(
    questions: list[dict],
    questions_dir: Path,
    split: str,
) -> None:
    out_path = questions_dir / f"CLEVR_{split}_questions.json"
    with open(out_path, "w") as f:
        json.dump({"questions": questions}, f)
    print(f"[questions] Saved {len(questions)} questions -> {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Download a mini-CLEVR subset.")
    parser.add_argument("--data-dir", default="data/clevr", help="Output directory")
    parser.add_argument("--n-train", type=int, default=5000,
                        help="Number of training images to download")
    parser.add_argument("--n-val", type=int, default=1000,
                        help="Number of validation images to download")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    questions_dir = data_dir / "questions"

    # 1. Questions
    download_questions(data_dir)

    # 2. Train split
    for split, n in [("train", args.n_train), ("val", args.n_val)]:
        raw_path = questions_dir / f"CLEVR_{split}_questions.json"
        if not raw_path.exists():
            print(f"[{split}] Questions file not found: {raw_path}")
            continue

        filtered_qs, image_fnames = filter_questions(raw_path, n)
        save_filtered_questions(filtered_qs, questions_dir, split)
        download_images(image_fnames, split, data_dir / "images" / split)

    print("\nDone! Dataset ready in", data_dir)


if __name__ == "__main__":
    main()
