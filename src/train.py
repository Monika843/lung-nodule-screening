"""
Training script.

Trains the DenseNet121 classifier on the SYNTHETIC demo dataset with a
PATIENT-LEVEL split (no leakage), and computes the full evaluation suite
requested: accuracy, precision, recall/sensitivity, specificity, F1,
ROC-AUC, confusion matrix.

To retrain on real LUNA16 patches: replace the `generate_dataset()` call
with your real patch-loading function that returns the same
(images, labels, patient_ids) shapes. Nothing else in this file changes.
"""
import sys
import os
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix, roc_curve)

sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import normalize_for_model
from synthetic_data import generate_dataset, patient_level_split
from model import build_model

DEVICE = torch.device("cpu")
IMG_SIZE = 64
BATCH_SIZE = 16
EPOCHS = 6
LR = 1e-4


class PatchDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = normalize_for_model(self.images[idx])          # 0-1 float32, HxW
        img = torch.from_numpy(img).unsqueeze(0)              # 1xHxW
        label = int(self.labels[idx])
        return img, label


def evaluate(model, loader):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(DEVICE)
            out = model(imgs)
            probs = torch.softmax(out, dim=1)[:, 1].cpu().numpy()
            preds = out.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            all_probs.extend(probs)

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_probs = np.array(all_probs)

    cm = confusion_matrix(all_labels, all_preds)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, zero_division=0),
        "sensitivity_recall": recall_score(all_labels, all_preds, zero_division=0),
        "specificity": specificity,
        "f1_score": f1_score(all_labels, all_preds, zero_division=0),
        "roc_auc": roc_auc_score(all_labels, all_probs) if len(set(all_labels)) > 1 else None,
        "confusion_matrix": cm.tolist(),
    }
    fpr, tpr, _ = roc_curve(all_labels, all_probs)
    return metrics, fpr.tolist(), tpr.tolist()


def main():
    print("Generating synthetic demo dataset...")
    images, labels, patient_ids = generate_dataset(n_per_class=150, size=IMG_SIZE)
    train_mask, val_mask, test_mask = patient_level_split(patient_ids)

    print(f"Total samples: {len(images)} | train patients: {len(set(patient_ids[train_mask]))}, "
          f"val patients: {len(set(patient_ids[val_mask]))}, test patients: {len(set(patient_ids[test_mask]))}")

    train_ds = PatchDataset(images[train_mask], labels[train_mask])
    val_ds = PatchDataset(images[val_mask], labels[val_mask])
    test_ds = PatchDataset(images[test_mask], labels[test_mask])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE)

    print("Building DenseNet121 (transfer learning)...")
    model = build_model(pretrained=True).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_accuracy": []}

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for imgs, labels_b in train_loader:
            imgs, labels_b = imgs.to(DEVICE), labels_b.to(DEVICE)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels_b)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * imgs.size(0)

        avg_loss = total_loss / len(train_ds)
        val_metrics, _, _ = evaluate(model, val_loader)
        history["train_loss"].append(avg_loss)
        history["val_accuracy"].append(val_metrics["accuracy"])
        print(f"Epoch {epoch}/{EPOCHS} | train_loss={avg_loss:.4f} | val_acc={val_metrics['accuracy']:.3f}")

    print("\nFinal evaluation on held-out TEST patients (never seen in training):")
    test_metrics, fpr, tpr = evaluate(model, test_loader)
    for k, v in test_metrics.items():
        print(f"  {k}: {v}")

    os.makedirs("../models", exist_ok=True)
    torch.save(model.state_dict(), "../models/densenet121_demo.pth")

    results = {
        "history": history,
        "test_metrics": test_metrics,
        "roc_fpr": fpr,
        "roc_tpr": tpr,
        "note": "Trained on SYNTHETIC demo data. Metrics reflect the demo "
                "task difficulty, not real clinical performance."
    }
    with open("../models/training_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved model -> ../models/densenet121_demo.pth")
    print("Saved metrics -> ../models/training_results.json")


if __name__ == "__main__":
    main()
