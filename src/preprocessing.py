"""
CT preprocessing utilities.

These functions implement the REAL preprocessing steps used for lung CT
(HU windowing, normalization) and work on any numpy array of Hounsfield
Unit values -- whether it comes from a real DICOM/.mhd scan or from the
synthetic demo generator in synthetic_data.py.

When you plug in real LUNA16/LIDC-IDRI data later, these functions do not
need to change.
"""
import numpy as np

# Standard lung window in Hounsfield Units (HU)
LUNG_WINDOW_MIN = -1000
LUNG_WINDOW_MAX = 400


def apply_lung_window(hu_slice: np.ndarray,
                       window_min: int = LUNG_WINDOW_MIN,
                       window_max: int = LUNG_WINDOW_MAX) -> np.ndarray:
    """Clip a Hounsfield-Unit CT slice to the standard lung window."""
    clipped = np.clip(hu_slice, window_min, window_max)
    return clipped


def normalize_to_uint8(hu_slice: np.ndarray,
                        window_min: int = LUNG_WINDOW_MIN,
                        window_max: int = LUNG_WINDOW_MAX) -> np.ndarray:
    """Scale a windowed HU slice to 0-255 uint8 for CNN input / display."""
    windowed = apply_lung_window(hu_slice, window_min, window_max)
    scaled = (windowed - window_min) / (window_max - window_min)
    scaled = np.clip(scaled, 0.0, 1.0)
    return (scaled * 255).astype(np.uint8)


def normalize_for_model(hu_slice: np.ndarray) -> np.ndarray:
    """Scale to 0-1 float32 for direct model input (no uint8 rounding)."""
    windowed = apply_lung_window(hu_slice)
    scaled = (windowed - LUNG_WINDOW_MIN) / (LUNG_WINDOW_MAX - LUNG_WINDOW_MIN)
    return np.clip(scaled, 0.0, 1.0).astype(np.float32)
