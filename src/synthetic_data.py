"""
SYNTHETIC DEMO DATA GENERATOR
==============================
This is NOT real patient data. It exists so the full pipeline
(preprocessing -> model -> Grad-CAM -> dashboard -> report) can be built,
run, and demoed TODAY without a multi-hour/multi-GB dataset download.

It generates grayscale patches that mimic HU-windowed lung CT crops:
  - "background" texture similar to lung parenchyma (noisy, low intensity)
  - class 0 (no_nodule): background texture only
  - class 1 (nodule_like): background + a small bright, roughly round blob,
    similar in appearance (not medical validity) to how a solid pulmonary
    nodule appears in a lung-windowed CT crop

REPLACING WITH REAL DATA LATER:
Swap `generate_dataset()` for a function that reads real cropped patches
from LUNA16 candidates.csv (crop 32-64px around each candidate x,y,z using
SimpleITK, using the SAME preprocessing.py functions). Keep the same
(image, label, patient_id) return shape and nothing downstream needs to
change. See README.md "Moving to Real Data" section for exact steps.
"""
import numpy as np


def _make_background(size: int, rng: np.random.Generator) -> np.ndarray:
    base = rng.normal(loc=-750, scale=60, size=(size, size))  # HU-like lung tissue noise
    # a few faint vessel-like streaks
    for _ in range(rng.integers(1, 4)):
        x0, y0 = rng.integers(0, size, size=2)
        length = rng.integers(size // 4, size // 2)
        angle = rng.uniform(0, 2 * np.pi)
        for t in range(length):
            x = int(x0 + t * np.cos(angle))
            y = int(y0 + t * np.sin(angle))
            if 0 <= x < size and 0 <= y < size:
                base[y, x] += 150
    return base


def _add_nodule(img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    size = img.shape[0]
    cx, cy = rng.integers(size // 3, 2 * size // 3, size=2)
    radius = rng.integers(3, size // 6)
    intensity = rng.uniform(150, 400)  # brighter, solid-tissue-like HU
    yy, xx = np.ogrid[:size, :size]
    mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
    img = img.copy()
    img[mask] += intensity
    return img


def generate_dataset(n_per_class: int = 150, size: int = 64, seed: int = 42,
                      n_patients: int = 40):
    """
    Returns:
        images: (N, size, size) float32 array of raw HU-like values
        labels: (N,) int array, 0 = no_nodule, 1 = nodule_like
        patient_ids: (N,) array of synthetic patient IDs (for patient-level split demo)
    """
    rng = np.random.default_rng(seed)
    images, labels, patient_ids = [], [], []

    for cls in (0, 1):
        for i in range(n_per_class):
            bg = _make_background(size, rng)
            img = _add_nodule(bg, rng) if cls == 1 else bg
            images.append(img.astype(np.float32))
            labels.append(cls)
            patient_ids.append(f"SYN{(i % n_patients):03d}")

    images = np.stack(images)
    labels = np.array(labels)
    patient_ids = np.array(patient_ids)

    # shuffle
    idx = rng.permutation(len(images))
    return images[idx], labels[idx], patient_ids[idx]


def patient_level_split(patient_ids: np.ndarray, seed: int = 42,
                         train_frac: float = 0.7, val_frac: float = 0.15):
    """
    Splits by PATIENT ID, not by sample -- this is the data-leakage-prevention
    step your brief specifically called out. All slices/patches from one
    patient end up in exactly one split.
    """
    rng = np.random.default_rng(seed)
    unique_patients = np.unique(patient_ids)
    rng.shuffle(unique_patients)

    n = len(unique_patients)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_p = set(unique_patients[:n_train])
    val_p = set(unique_patients[n_train:n_train + n_val])
    test_p = set(unique_patients[n_train + n_val:])

    train_mask = np.array([p in train_p for p in patient_ids])
    val_mask = np.array([p in val_p for p in patient_ids])
    test_mask = np.array([p in test_p for p in patient_ids])
    return train_mask, val_mask, test_mask
