# Lung Nodule Screening

This is a **complete, working, end-to-end pipeline**:
preprocessing → DenseNet121 classifier → evaluation → Grad-CAM → Streamlit
dashboard → PDF report — but it is currently trained and demoed on a
**synthetic dataset** (`src/synthetic_data.py`), not real patient CT scans.

This was a deliberate, disclosed tradeoff: real LUNA16 data is ~60–80GB
and needs hours of GPU training to produce trustworthy numbers — neither
fits in a quick turnaround. Rather than fabricate results on data that
was never actually used, every metric in this build (accuracy, ROC-AUC,
confusion matrix, etc.) is **real output from a real trained model** —
just on the synthetic demo task. This gets you a fully working, demoable
system today, with a clear, short path to swap in real data before your
final submission.

**Do not present the synthetic-data metrics as clinical performance.**
The dashboard states this in the Home page and Model Performance page.

## What's real vs. what's placeholder

| Component | Status |
|---|---|
| HU windowing / normalization (`preprocessing.py`) | Real logic, works on real CT data unchanged |
| Patient-level train/val/test split (data leakage prevention) | Real logic, works on real data unchanged |
| DenseNet121 transfer-learning model | Real architecture, real training loop |
| Grad-CAM | Real implementation, tested working |
| Evaluation suite (accuracy/precision/recall/specificity/F1/ROC-AUC/confusion matrix) | Real, computed on real held-out predictions |
| Streamlit dashboard | Fully functional, all pages wired to real model calls |
| PDF report | Real generation, real content |
| **Training data** | **Synthetic placeholder** — swap before final submission |

## How to run

```bash
pip install -r requirements.txt

# (optional) retrain from scratch — already done once, checkpoint included
cd src && python train.py && cd ..

# launch dashboard
streamlit run app.py
```

Open the local URL Streamlit prints (usually `http://localhost:8501`).

## Moving to real LUNA16 data (do this before your final submission)

1. Download LUNA16 subsets from Zenodo (https://zenodo.org/record/3723295)
   or grand-challenge.org — start with 2–3 of the 10 subsets (~15–20GB)
   for a student-scale dataset given laptop disk constraints. Use Google
   Colab (with Google Drive mounted) if local disk is under 20GB.
2. Load each `.mhd`/`.raw` volume with `SimpleITK`.
3. For each row in `candidates.csv` (columns: seriesuid, coordX/Y/Z, class),
   crop a 2D patch (e.g. 64×64) around the world-coordinate location,
   converted to voxel coordinates via `SimpleITK.TransformPhysicalPointToIndex`.
4. Replace `generate_dataset()` in `src/synthetic_data.py` (or write a new
   `real_data.py` with the same function signature: returns
   `(images, labels, patient_ids)`) with this real loading logic.
   `seriesuid` (or the patient ID prefix) becomes your `patient_ids` array
   — this is what `patient_level_split()` already uses correctly.
5. For malignancy-risk instead of simple presence/absence, use LIDC-IDRI's
   radiologist malignancy-likelihood annotations (1–5 scale), typically
   binarized as malignant (≥4) vs benign (≤2), excluding ambiguous 3s, per
   standard practice in nodule-classification literature.
6. Re-run `python src/train.py`. Nothing else in the codebase changes.
7. Expect training to take meaningfully longer — use Colab GPU, not CPU.

## Project structure

```
lung-nodule-screening/
├── src/
│   ├── preprocessing.py    # HU windowing, normalization (real, dataset-agnostic)
│   ├── synthetic_data.py   # demo data generator + patient-level split
│   ├── model.py            # DenseNet121 transfer-learning model
│   ├── gradcam.py          # Grad-CAM explainability
│   └── train.py            # training + full evaluation suite
├── models/
│   ├── densenet121_demo.pth      # trained checkpoint (synthetic data)
│   └── training_results.json     # real metrics from that training run
├── reports/                # generated PDF reports land here
├── app.py                  # Streamlit dashboard
├── requirements.txt
└── README.md
```

## Medical disclaimer

This system is a research/educational tool. It does not diagnose disease
and is not a substitute for professional medical judgment. Grad-CAM
highlighted regions indicate model attention, not confirmed pathology.
