"""Mini-CLEVR Dataset — supports both raw images and pre-extracted features.

Two modes
---------
1. **Feature mode** (recommended, matches the paper):
   Loads pre-extracted ResNet101 features from HDF5 files produced by
   `data/extract_features.py`. Features have shape (1024, 14, 14).

2. **Image mode** (fallback, lower accuracy):
   Loads raw PNG images and applies resize + normalise transforms.

The mode is selected automatically: if `data_dir/features_{split}.h5` exists,
feature mode is used; otherwise raw images are loaded.

Answer vocabulary
-----------------
28 canonical CLEVR answers — same set as the original paper.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import h5py
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


# ── Answer vocabulary ────────────────────────────────────────────────────────

CLEVR_ANSWERS = [
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10",
    "yes", "no",
    "red", "blue", "green", "yellow", "cyan", "purple", "gray", "brown",
    "rubber", "metal",
    "small", "large",
    "sphere", "cube", "cylinder",
]
ANSWER_TO_IDX: dict[str, int] = {a: i for i, a in enumerate(CLEVR_ANSWERS)}
NUM_ANSWERS = len(CLEVR_ANSWERS)  # 28


# ── Vocabulary helpers ────────────────────────────────────────────────────────

def build_vocab(questions: list[dict]) -> dict[str, int]:
    """Build word→index mapping; index 0 = <PAD>, 1 = <UNK>."""
    vocab: dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
    for q in questions:
        for word in q["question"].lower().split():
            word = word.strip("?;,.")
            if word not in vocab:
                vocab[word] = len(vocab)
    return vocab


# ── Dataset ───────────────────────────────────────────────────────────────────

class MiniCLEVR(Dataset):
    """CLEVR dataset with automatic feature / raw-image mode selection.

    Parameters
    ----------
    data_dir : path to CLEVR root (contains images/ and questions/)
    split    : "train" | "val"
    max_samples : max number of samples to load
    img_size : image resize target (only used in raw-image mode)
    max_question_length : pad/truncate questions to this length
    vocab : reuse an existing word vocab (e.g. from training split)
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: str = "train",
        max_samples: int = 5000,
        img_size: int = 128,
        max_question_length: int = 45,
        vocab: Optional[dict[str, int]] = None,
    ) -> None:
        super().__init__()
        self.data_dir = Path(data_dir)
        self.split = split
        self.img_size = img_size
        self.max_question_length = max_question_length

        # ── Questions JSON ──────────────────────────────────────────────────
        questions_path = self.data_dir / "questions" / f"CLEVR_{split}_questions.json"
        with open(questions_path) as f:
            all_questions = json.load(f)["questions"]
        self.samples = all_questions[:max_samples]

        # ── Vocabulary ──────────────────────────────────────────────────────
        self.vocab = vocab if vocab is not None else build_vocab(self.samples)
        self.vocab_size = len(self.vocab)

        # ── Feature mode detection ──────────────────────────────────────────
        h5_path = self.data_dir / f"features_{split}.h5"
        self.use_features = h5_path.exists()

        if self.use_features:
            self._h5_path = h5_path
            self._h5_file = None   # opened lazily (required for multi-worker)
            # Build filename → h5 row index mapping
            with h5py.File(h5_path, "r") as f:
                image_ids = [s.decode() if isinstance(s, bytes) else s
                             for s in f["image_ids"][:]]
            self._fname_to_idx: dict[str, int] = {
                name: i for i, name in enumerate(image_ids)
            }
            # Determine feature shape
            with h5py.File(h5_path, "r") as f:
                self.feature_shape = tuple(f["features"].shape[1:])  # (C, H, W)
        else:
            self.feature_shape = (3, img_size, img_size)
            self._transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])
            self._image_dir = self.data_dir / "images" / split

    # ── Lazy H5 open (supports DataLoader workers) ───────────────────────────

    def _get_h5(self) -> h5py.File:
        if self._h5_file is None:
            self._h5_file = h5py.File(self._h5_path, "r")
        return self._h5_file

    def close(self) -> None:
        if self._h5_file is not None:
            self._h5_file.close()
            self._h5_file = None

    def __del__(self) -> None:
        self.close()

    # ── Encoding helpers ─────────────────────────────────────────────────────

    def encode_question(self, question_str: str) -> torch.Tensor:
        unk = self.vocab.get("<UNK>", 1)
        indices = [self.vocab.get(w.strip("?;,."), unk)
                   for w in question_str.lower().split()]
        indices = indices[: self.max_question_length]
        indices += [0] * (self.max_question_length - len(indices))
        return torch.tensor(indices, dtype=torch.long)

    @staticmethod
    def encode_answer(answer_str: str) -> int:
        return ANSWER_TO_IDX.get(answer_str.lower(), -1)

    # ── Dataset interface ────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        item = self.samples[idx]
        fname = item["image_filename"]

        # Visual input
        if self.use_features:
            h5 = self._get_h5()
            row = self._fname_to_idx[fname]
            visual = torch.from_numpy(h5["features"][row])  # (C, H, W) float32
        else:
            img_path = self._image_dir / fname
            visual = self._transform(Image.open(img_path).convert("RGB"))

        question = self.encode_question(item["question"])
        answer   = self.encode_answer(item["answer"])

        return visual, question, answer, item["question"], item["answer"]
