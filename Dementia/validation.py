#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CV on connectome npy (3-class: NC/AD/LBD)
Models: SVM, Logistic Regression, XGBoost, GCN
CV: 5/10-fold, stratified by site×label
Features: classic models use upper-triangular vectorization of connectivity
Selection: optional XGBoost-based RFE
Grid search: per model
Outputs:
  - Per-fold metrics & best params (CSV)
  - Overall metrics (CSV)
  - Out-of-fold predictions for t-test (CSV) with per-subject probs/preds
  - Feature masks when RFE enabled (npz)

Assumptions about labels in npy:
- original labels: 0=NC, 1=MCI, 2=AD, 3=LBD
- THIS SCRIPT filters to {0,2,3} and remaps to contiguous 0..2:
    map {0->0 (NC), 2->1 (AD), 3->2 (LBD)}
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from itertools import product
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score, f1_score, balanced_accuracy_score, confusion_matrix, recall_score
)
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, TransformerMixin
import warnings

# XGBoost
try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False
    warnings.warn("xgboost not available; XGB-related options will be skipped.", RuntimeWarning)

# Torch for GCN
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False
    warnings.warn("PyTorch not available; GCN will be skipped.", RuntimeWarning)


# ---------------------- Data loading & feature building ----------------------

CLASS_ORDER = ["NC", "AD", "LBD"]           # fixed order for probabilities/metrics
CLASS_MAP_INTO_3 = {0: 0, 2: 1, 3: 2}      # 0->NC, 2->AD, 3->LBD

def load_npy_filter_remap(npy_path: Path):
    """
    Load list of dicts, keep only NC/AD/LBD, and remap labels to {0,1,2}=[NC,AD,LBD].
    """
    arr = np.load(npy_path, allow_pickle=True)
    recs = []
    for r in arr:
        lab = r.get("label", None)
        if lab in CLASS_MAP_INTO_3:
            r = dict(r)  # copy
            r["label_orig"] = lab
            r["label"] = CLASS_MAP_INTO_3[lab]
            recs.append(r)
    if len(recs) == 0:
        raise RuntimeError("No subjects left after filtering to NC/AD/LBD.")
    return recs

def upper_triangle_vector(mat: np.ndarray) -> np.ndarray:
    """Vectorize upper triangle (exclude diagonal)."""
    n = mat.shape[0]
    iu = np.triu_indices(n, k=1)
    return mat[iu]

def stack_features_upper(recs):
    """Return X (subjects × features), y (labels 0..2), meta DataFrame, plus tri index."""
    n = recs[0]["matrix"].shape[0]
    iu = np.triu_indices(n, k=1)
    X = np.vstack([upper_triangle_vector(r["matrix"]) for r in recs])
    y = np.array([int(r["label"]) for r in recs], dtype=int)
    meta = pd.DataFrame({
        "subject_id": [r["subject_id"] for r in recs],
        "site": [r["site"] for r in recs],
        "age": [r.get("age", None) for r in recs],
        "gender": [r.get("gender", None) for r in recs],
        "mmse": [r.get("mmse", None) for r in recs],
        "roi_n": n
    })
    return X, y, meta, iu, n


# ---------------------- Metrics ----------------------

def macro_auc(y_true, proba, n_classes=3):
    """
    Macro AUC in multiclass (one-vs-rest). If some class is absent in y_true of a fold,
    we skip it for that fold (to avoid undefined AUC).
    """
    aucs = []
    for c in range(n_classes):
        y_bin = (y_true == c).astype(int)
        if y_bin.sum() == 0 or y_bin.sum() == len(y_bin):
            continue
        aucs.append(roc_auc_score(y_bin, proba[:, c]))
    return float(np.mean(aucs)) if aucs else np.nan

def macro_specificity(y_true, y_pred, n_classes=3):
    """
    Macro specificity: average over classes of TN / (TN + FP) treating each class as positive in turn.
    """
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))
    specs = []
    for i in range(n_classes):
        TP = cm[i, i]
        FN = cm[i, :].sum() - TP
        FP = cm[:, i].sum() - TP
        TN = cm.sum() - TP - FN - FP
        denom = (TN + FP)
        if denom == 0:
            continue
        specs.append(TN / denom)
    return float(np.mean(specs)) if specs else np.nan

def compute_metrics(y_true, y_pred, proba, n_classes=3):
    return {
        "auc_macro": macro_auc(y_true, proba, n_classes=n_classes),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "sensitivity_macro": recall_score(y_true, y_pred, average="macro"),
        "specificity_macro": macro_specificity(y_true, y_pred, n_classes=n_classes),
    }


# ---------------------- XGBoost-based RFE ----------------------

class XGBRFE(BaseEstimator, TransformerMixin):
    """
    Recursive Feature Elimination using an XGBClassifier as estimator.
    """
    def __init__(self, n_features_to_select=500, step=0.1, xgb_params=None, random_state=42):
        self.n_features_to_select = n_features_to_select
        self.step = step
        self.xgb_params = xgb_params or {}
        self.random_state = random_state
        self.support_mask_ = None

    def fit(self, X, y):
        if not _HAS_XGB:
            raise RuntimeError("xgboost not available for XGBRFE.")
        X = np.asarray(X)
        n_feats = X.shape[1]
        idx = np.arange(n_feats)

        params = dict(self.xgb_params)
        params.setdefault("random_state", self.random_state)
        params.setdefault("n_estimators", 200)
        params.setdefault("max_depth", 4)
        params.setdefault("learning_rate", 0.1)
        params.setdefault("subsample", 0.8)
        params.setdefault("colsample_bytree", 0.8)
        params.setdefault("n_jobs", 1)
        params.setdefault("objective", "multi:softprob")
        params.setdefault("num_class", 3)
        params.setdefault("eval_metric", "mlogloss")

        clf = XGBClassifier(**params)

        current = idx.copy()
        while len(current) > self.n_features_to_select:
            clf.fit(X[:, current], y)
            importances = clf.feature_importances_
            if importances is None or np.all(importances == 0):
                ranks = np.arange(len(current))
            else:
                ranks = np.argsort(importances)  # least important first

            if self.step >= 1:
                remove_k = int(self.step)
            else:
                remove_k = max(1, int(np.ceil(self.step * len(current))))
            remove_k = min(remove_k, len(current) - self.n_features_to_select)
            remove_idx_local = ranks[:remove_k]
            keep_mask_local = np.ones(len(current), dtype=bool)
            keep_mask_local[remove_idx_local] = False
            current = current[keep_mask_local]

        self.support_mask_ = np.zeros(n_feats, dtype=bool)
        self.support_mask_[current] = True
        return self

    def transform(self, X):
        if self.support_mask_ is None:
            raise RuntimeError("Call fit before transform.")
        return np.asarray(X)[:, self.support_mask_]

    def get_support(self, indices=False):
        if indices:
            return np.where(self.support_mask_)[0]
        return self.support_mask_


# ---------------------- GCN (simple reference) ----------------------

if _HAS_TORCH:
    class ConnectomeDataset(Dataset):
        def __init__(self, mats, labels):
            self.mats = mats
            self.labels = labels
        def __len__(self): return len(self.mats)
        def __getitem__(self, i):
            A = self.mats[i].astype(np.float32)
            y = int(self.labels[i])
            return torch.from_numpy(A), torch.tensor(y, dtype=torch.long)

    class GCNLayer(nn.Module):
        def __init__(self, in_dim, out_dim):
            super().__init__()
            self.lin = nn.Linear(in_dim, out_dim, bias=False)
        def forward(self, X, A):
            I = torch.eye(A.shape[-1], device=A.device)
            A_hat = A + I
            deg = A_hat.sum(-1)
            D_inv_sqrt = torch.diag_embed(torch.pow(deg, -0.5).clamp(min=1e-6))
            A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt
            return A_norm @ self.lin(X)

    class SimpleGCN(nn.Module):
        def __init__(self, n_nodes, hidden=64, num_classes=3, dropout=0.2):
            super().__init__()
            self.gcn1 = GCNLayer(n_nodes, hidden)
            self.gcn2 = GCNLayer(hidden, hidden)
            self.dropout = nn.Dropout(dropout)
            self.cls = nn.Linear(hidden, num_classes)
        def forward(self, A):
            X = A
            H = torch.relu(self.gcn1(X, A))
            H = self.dropout(H)
            H = torch.relu(self.gcn2(H, A))
            H = self.dropout(H)
            Z = H.mean(dim=1)
            logits = self.cls(Z)
            return logits

    @torch.no_grad()
    def gcn_predict(model, mats, batch_size=16, device="cpu"):
        ds = ConnectomeDataset(mats, np.zeros(len(mats), dtype=int))
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False)
        model.eval()
        all_logits = []
        for A, _ in dl:
            A = A.to(device)
            logits = model(A)
            all_logits.append(logits.cpu())
        logits = torch.cat(all_logits, dim=0)
        probs = torch.softmax(logits, dim=1).numpy()
        preds = probs.argmax(1)
        return preds, probs


# ---------------------- Training routines ----------------------

def site_label_strat(meta: pd.DataFrame, y: np.ndarray):
    """Stratify by combined key to preserve site and class distribution."""
    return np.array([f"{s}|{l}" for s, l in zip(meta["site"].values, y)])

def run_classic_models(X, y, meta, folds, out_dir, use_rfe, models):
    out_dir.mkdir(parents=True, exist_ok=True)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1234)
    strat_key = site_label_strat(meta, y)

    grids = {}
    if "svm" in models:
        grids["svm"] = {
            "clf__kernel": ["linear", "rbf"],
            "clf__C": [0.1, 1, 10],
            "clf__gamma": ["scale", 0.01, 0.001]
        }
    if "logreg" in models:
        grids["logreg"] = {
            "clf__C": [0.1, 1, 10],
            "clf__penalty": ["l2"],
            "clf__solver": ["lbfgs", "liblinear", "saga"],
            "clf__max_iter": [2000],
        }
    if "xgb" in models and _HAS_XGB:
        grids["xgb"] = {
            "clf__n_estimators": [200, 400],
            "clf__max_depth": [3, 5],
            "clf__learning_rate": [0.1, 0.05],
            "clf__subsample": [0.8, 1.0],
            "clf__colsample_bytree": [0.8, 1.0],
            "clf__reg_lambda": [1.0, 5.0]
        }

    results_fold = []
    oof_rows = []
    feature_masks = {}

    for fold_idx, (tr, te) in enumerate(skf.split(X, strat_key), start=1):
        X_tr, X_te = X[tr], X[te]
        y_tr, y_te = y[tr], y[te]
        meta_te = meta.iloc[te].reset_index(drop=True)

        for model_name in models:
            if model_name == "xgb" and not _HAS_XGB:
                continue

            steps = [("scaler", StandardScaler(with_mean=True, with_std=True))]

            if use_rfe:
                rfe = XGBRFE(
                    n_features_to_select=min(1000, max(50, X.shape[1] // 2)),
                    step=0.1,
                    xgb_params={"n_estimators": 200, "max_depth": 3, "learning_rate": 0.1,
                                "objective": "multi:softprob", "num_class": 3, "eval_metric": "mlogloss"},
                    random_state=fold_idx
                )
                steps.append(("rfe", rfe))

            if model_name == "svm":
                clf = SVC(probability=True, class_weight="balanced", random_state=fold_idx)
            elif model_name == "logreg":
                clf = LogisticRegression(multi_class="auto", class_weight="balanced", max_iter=2000, n_jobs=None)
            elif model_name == "xgb":
                clf = XGBClassifier(objective="multi:softprob", num_class=3, n_jobs=1,
                                    eval_metric="mlogloss", random_state=fold_idx)
            else:
                continue

            steps.append(("clf", clf))
            pipe = Pipeline(steps)

            param_grid = grids.get(model_name, {})
            gs = GridSearchCV(
                pipe, param_grid=param_grid,
                scoring="balanced_accuracy", cv=3, n_jobs=1, refit=True
            )
            gs.fit(X_tr, y_tr)

            best = gs.best_estimator_
            if use_rfe and "rfe" in dict(best.named_steps):
                mask = best.named_steps["rfe"].get_support(indices=False)
                feature_masks[(model_name, fold_idx)] = mask

            y_prob = best.predict_proba(X_te)
            y_pred = y_prob.argmax(1)
            metrics = compute_metrics(y_te, y_pred, y_prob, n_classes=3)
            results_fold.append({
                "fold": fold_idx,
                "model": model_name,
                "n_test": len(y_te),
                "best_params": str(gs.best_params_),
                **metrics
            })

            for i in range(len(y_te)):
                oof_rows.append({
                    "fold": fold_idx,
                    "model": model_name,
                    "subject_id": meta_te.loc[i, "subject_id"],
                    "site": meta_te.loc[i, "site"],
                    "true": int(y_te[i]),
                    "pred": int(y_pred[i]),
                    "p_nc": y_prob[i, 0],
                    "p_ad": y_prob[i, 1],
                    "p_lbd": y_prob[i, 2],
                })

    res_df = pd.DataFrame(results_fold)
    oof_df = pd.DataFrame(oof_rows)
    res_df.to_csv(out_dir / "cv_fold_metrics.csv", index=False)
    agg = res_df.groupby(["model"]).agg(
        auc_macro=("auc_macro", "mean"),
        f1_macro=("f1_macro", "mean"),
        sensitivity_macro=("sensitivity_macro", "mean"),
        specificity_macro=("specificity_macro", "mean"),
        balanced_acc=("balanced_acc", "mean")
    ).reset_index()
    agg.to_csv(out_dir / "cv_overall_metrics.csv", index=False)
    oof_df.to_csv(out_dir / "oof_predictions.csv", index=False)
    if feature_masks:
        np.savez(out_dir / "rfe_feature_masks.npz", **{f"{k[0]}_fold{k[1]}": v for k, v in feature_masks.items()})

def run_gcn(recs, y, meta, folds, out_dir):
    if not _HAS_TORCH:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=1234)
    strat_key = site_label_strat(meta, y)

    mats = np.stack([r["matrix"] for r in recs])  # (S, N, N)
    n_nodes = mats.shape[1]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    grid = list(product([32, 64], [1e-3, 5e-4], [0.0, 1e-4]))  # (hidden, lr, weight_decay)

    results_fold = []
    oof_rows = []
    fold_idx = 0

    for tr, te in skf.split(mats, strat_key):
        fold_idx += 1
        A_tr, A_te = mats[tr], mats[te]
        y_tr, y_te = y[tr], y[te]
        meta_te = meta.iloc[te].reset_index(drop=True)

        best_score = -np.inf
        best_state = None
        best_cfg = None

        for hidden, lr, wd in grid:
            model = SimpleGCN(n_nodes=n_nodes, hidden=hidden, num_classes=3, dropout=0.2).to(device)
            opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
            ds_tr = ConnectomeDataset(A_tr, y_tr)
            dl_tr = DataLoader(ds_tr, batch_size=16, shuffle=True)
            for _ in range(20):
                model.train()
                for A, yb in dl_tr:
                    A = A.to(device)
                    yb = yb.to(device)
                    opt.zero_grad()
                    logits = model(A)
                    loss = F.cross_entropy(logits, yb)
                    loss.backward()
                    opt.step()
            preds_tr, probs_tr = gcn_predict(model, A_tr, batch_size=32, device=device)
            score = balanced_accuracy_score(y_tr, preds_tr)
            if score > best_score:
                best_score = score
                best_state = model.state_dict()
                best_cfg = (hidden, lr, wd)

        model = SimpleGCN(n_nodes=n_nodes, hidden=best_cfg[0], num_classes=3, dropout=0.2).to(device)
        model.load_state_dict(best_state)
        y_pred, y_prob = gcn_predict(model, A_te, batch_size=32, device=device)
        metrics = compute_metrics(y_te, y_pred, y_prob, n_classes=3)

        results_fold.append({
            "fold": fold_idx,
            "model": "gcn",
            "n_test": len(y_te),
            "best_params": f"hidden={best_cfg[0]}, lr={best_cfg[1]}, wd={best_cfg[2]}",
            **metrics
        })
        for i in range(len(y_te)):
            oof_rows.append({
                "fold": fold_idx,
                "model": "gcn",
                "subject_id": meta_te.loc[i, "subject_id"],
                "site": meta_te.loc[i, "site"],
                "true": int(y_te[i]),
                "pred": int(y_pred[i]),
                "p_nc": y_prob[i, 0],
                "p_ad": y_prob[i, 1],
                "p_lbd": y_prob[i, 2],
            })

    res_df = pd.DataFrame(results_fold)
    oof_df = pd.DataFrame(oof_rows)
    res_df.to_csv(out_dir / "cv_fold_metrics.csv", index=False)
    agg = res_df.groupby(["model"]).agg(
        auc_macro=("auc_macro", "mean"),
        f1_macro=("f1_macro", "mean"),
        sensitivity_macro=("sensitivity_macro", "mean"),
        specificity_macro=("specificity_macro", "mean"),
        balanced_acc=("balanced_acc", "mean")
    ).reset_index()
    agg.to_csv(out_dir / "cv_overall_metrics.csv", index=False)
    oof_df.to_csv(out_dir / "oof_predictions.csv", index=False)


# ---------------------- CLI ----------------------

def parse_args():
    ap = argparse.ArgumentParser(description="CV with SVM/LogReg/XGBoost/GCN on connectome npy (3-class NC/AD/LBD).")
    ap.add_argument("--npy", required=True, help="Path to merged npy (list of dicts).")
    ap.add_argument("--folds", type=int, default=5, choices=[5, 10], help="Number of CV folds.")
    ap.add_argument("--models", nargs="+", default=["svm","logreg","xgb","gcn"],
                    help="Subset of models to run. Any of: svm logreg xgb gcn")
    ap.add_argument("--rfe", type=int, default=0, help="Use XGBoost-based RFE in classic models (1=yes,0=no).")
    ap.add_argument("--out-dir", required=True, help="Output directory.")
    return ap.parse_args()

def main():
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    recs = load_npy_filter_remap(Path(args.npy))     # <-- filter to NC/AD/LBD and remap to 0..2
    X, y, meta, iu, n = stack_features_upper(recs)

    classic_models = [m for m in args.models if m in {"svm","logreg","xgb"}]
    if classic_models:
        run_classic_models(
            X=X, y=y, meta=meta, folds=args.folds,
            out_dir=out_dir / "classic",
            use_rfe=bool(args.rfe),
            models=classic_models
        )

    if "gcn" in args.models:
        if _HAS_TORCH:
            run_gcn(recs, y, meta, folds=args.folds, out_dir=out_dir / "gcn")
        else:
            print("[WARN] PyTorch not available; skip GCN.")

    print(f"[OK] All done. Results saved under: {out_dir}")

if __name__ == "__main__":
    main()
