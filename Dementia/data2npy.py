#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Merge per-subject connectivity (.txt) + ROI labels (_regions.tsv) with site metadata
from three Excel files (CAT, JT, MILOS) into a single .npy object array.

Each element in the saved npy is a dict:
{
  'subject_id': str,
  'site': str,              # 'CAT' | 'JT' | 'MILOS'
  'matrix': np.ndarray,     # (N,N) float64
  'regions': np.ndarray,    # (N,) ROI names
  'age': float or None,
  'gender': int or None,    # 0=male, 1=female, None=missing
  'mmse': float or None,
  'label': int              # 0=NC, 1=MCI, 2=AD, 3=LBD
}

Rules implemented (as provided by user):
- JT:
  * File names start with 'JT_' and subject IDs look like 'JT_D{number}A/B'.
  * In Excel, IDs end with 'A'. If filename ends with 'B', map to its 'A' twin for metadata.
  * Excel: Diagnosis: 1=AD, 2=CN, 3=LBD; Gender: 1=male, 2=female; has Age, MMSE.
- CAT:
  * File names start with 'CAT'.
  * Excel: subject id column 'CAT NO'; label column 'DIAGNOSIS'.
  * DIAGNOSIS: 0=HC(NC), 1=AD, 2=DLB(LBD), 3=PDD(LBD).
  * Has MMSE; no Age/Gender (left as None).
- MILOS:
  * File names are the remaining subjects (not starting with JT_ nor CAT).
  * Excel: match filenames to column 'WBIC ID' (but that column contains commas, e.g., '23,456').
    We strip commas to match file basename (e.g., '23456').
  * Participant ID is the subject ID (not used for matching).
  * Columns: Age, Gender with values like 'Male (1)' / 'Female (2)', no MMSE.
  * Group mapping: HC=NC, DLB/PDD=LBD, MCI-AD/MCI-LB=MCI, AD=AD.
    If none of those exact, do fuzzy: contains 'AD'→AD, contains 'DLB'/'PDD'/'LB'→LBD.
- Unified coding:
  * gender: 0=male, 1=female
  * label:  0=NC, 1=MCI, 2=AD, 3=LBD
- Subjects with missing/unresolvable label are skipped entirely.
- Statistics: per-site and overall, per group:
  counts, age mean±std (ignoring NaN), and gender counts.

Author: Minheng Chen
Date: 10/25/2025
python data2npy.py --mat-dir /mnt/raid/data_cambridge/MIND_DKT--out-npy DKT.npy; 
"""

import argparse
import os
from pathlib import Path
import re
import numpy as np
import pandas as pd

# -------------------------- Helpers: site detection & id normalization --------------------------

def detect_site(subject_id: str) -> str:
    if subject_id.startswith("JT_"):
        return "JT"
    if subject_id.startswith("CAT"):
        return "CAT"
    return "MILOS"

def normalize_jt_id_for_lookup(subj: str) -> str:
    """
    JT filenames look like 'JT_D123A' or 'JT_D123B'. Excel IDs end with 'A'.
    If endswith 'B', convert to 'A' for lookup.
    """
    if subj.endswith("B"):
        return subj[:-1] + "A"
    return subj

def normalize_milos_wbic(s: str) -> str:
    """Strip commas and spaces from WBIC ID to match filename (e.g., '23,456' -> '23456')."""
    if pd.isna(s):
        return ""
    return re.sub(r"[,\s]", "", str(s))

# -------------------------- Mapping functions --------------------------

def map_gender(value, site: str):
    """
    Map gender to unified coding: 0=male, 1=female, None if missing/unknown.
    JT: numeric 1=male, 2=female
    MILOS: string like 'Male (1)' or 'Female (2)' (case-insensitive)
    CAT: no gender -> None
    """
    if site == "JT":
        try:
            v = int(value)
            if v == 1: return 0
            if v == 2: return 1
        except Exception:
            pass
        return None
    elif site == "MILOS":
        if isinstance(value, str):
            low = value.strip().lower()
            if "male" in low: return 0
            if "female" in low: return 1
        # sometimes they might be plain numbers
        try:
            v = int(value)
            if v == 1: return 0
            if v == 2: return 1
        except Exception:
            pass
        return None
    elif site == "CAT":
        return None
    return None

def map_label(site: str, raw_label):
    """
    Unified label: 0=NC, 1=MCI, 2=AD, 3=LBD. Return None if cannot resolve.
    """
    if site == "JT":
        # Diagnosis: 1=AD, 2=CN, 3=LBD
        try:
            d = int(raw_label)
            if d == 1: return 2
            if d == 2: return 0
            if d == 3: return 3
        except Exception:
            return None
        return None

    if site == "CAT":
        # DIAGNOSIS: 0=HC(NC), 1=AD, 2=DLB(LBD), 3=PDD(LBD)
        try:
            d = int(raw_label)
            if d == 0: return 0
            if d == 1: return 2
            if d == 2: return 3
            if d == 3: return 3
        except Exception:
            return None
        return None

    if site == "MILOS":
        # Group messy: HC->NC, DLB/PDD->LBD, MCI-AD/MCI-LB->MCI, AD->AD
        # Else fuzzy: contains 'AD' -> AD; contains 'DLB'/'PDD'/'LB' -> LBD; contains 'MCI' -> MCI
        if pd.isna(raw_label):
            return None
        s = str(raw_label).strip().upper()
        exact_map = {
            "HC": 0, "NC": 0,
            "DLB": 3, "PDD": 3, "LBD": 3,
            "MCI-AD": 1, "MCI-LB": 1, "MCI": 1,
            "AD": 2
        }
        if s in exact_map:
            return exact_map[s]
        # fuzzy
        if "MCI" in s: return 1
        if "AD" in s: return 2
        if ("DLB" in s) or ("PDD" in s) or ("LB" in s): return 3
        if "HC" in s or "NC" in s: return 0
        return None

    return None

# -------------------------- Excel loaders --------------------------

def load_jt_xlsx(path: Path) -> pd.DataFrame:
    """
    Expect columns including: Subject ID (string like JT_D123A), Diagnosis (1/2/3),
    Gender (1/2), Age, MMSE. Column names might vary slightly; we select by fuzzy names.
    """
    df = pd.read_excel(path)
    # Find subject column: first column whose values look like JT_*
    subj_col = None
    for c in df.columns:
        vals = df[c].astype(str).str.startswith("JT_")
        if vals.mean() > 0.5:  # heuristic
            subj_col = c
            break
    if subj_col is None:
        # fallbacks
        for c in ["SubjectID", "Subject ID", "ID", "Subject"]:
            if c in df.columns:
                subj_col = c
                break
    if subj_col is None:
        raise ValueError(f"[JT] Cannot find subject ID column in {path}")

    # Normalize subject IDs to end with A for lookup mapping (source of truth in sheet)
    df["_subj_norm"] = df[subj_col].astype(str).str.replace(r"[ \t]", "", regex=True)
    df["_subj_norm"] = df["_subj_norm"].str.replace(r"B$", "A", regex=True)

    # Standardize column names (case-insensitive fuzzy)
    def pick(colnames, *cands):
        for k in colnames:
            for c in cands:
                if k.strip().lower() == c.strip().lower():
                    return k
        # loose contains
        for k in colnames:
            for c in cands:
                if c.strip().lower() in k.strip().lower():
                    return k
        return None

    diag_col = pick(df.columns, "Diagnosis")
    gender_col = pick(df.columns, "Gender")
    age_col = pick(df.columns, "Age")
    mmse_col = pick(df.columns, "MMSE")

    return df, subj_col, diag_col, gender_col, age_col, mmse_col

def load_cat_xlsx(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    if "CAT NO" not in df.columns:
        # fallback variants
        cand = None
        for c in df.columns:
            if "cat" in c.lower() and "no" in c.lower():
                cand = c; break
        if cand is None:
            raise ValueError(f"[CAT] Cannot find 'CAT NO' column in {path}")
        subj_col = cand
    else:
        subj_col = "CAT NO"
    # DIAGNOSIS
    diag_col = None
    for c in df.columns:
        if c.strip().lower() == "diagnosis":
            diag_col = c; break
    if diag_col is None:
        raise ValueError(f"[CAT] Cannot find 'DIAGNOSIS' column in {path}")
    # MMSE (if exists)
    mmse_col = None
    for c in df.columns:
        if "mmse" in c.strip().lower():
            mmse_col = c; break
    return df, subj_col, diag_col, mmse_col

def load_milos_xlsx(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path)
    # WBIC ID column: contains commas; we match filename by this (after stripping commas)
    w_col = None
    for c in df.columns:
        if "wbic" in c.lower() and "id" in c.lower():
            w_col = c; break
    if w_col is None:
        raise ValueError(f"[MILOS] Cannot find 'WBIC ID' column in {path}")

    # Participant ID (subject id)
    p_col = None
    for c in df.columns:
        if "participant" in c.lower() and "id" in c.lower():
            p_col = c; break
    if p_col is None:
        # not strictly required for matching, but we try to pick it
        for c in df.columns:
            if "participant" in c.lower():
                p_col = c; break

    # Other columns
    age_col = None
    for c in df.columns:
        if c.strip().lower() == "age":
            age_col = c; break
    gender_col = None
    for c in df.columns:
        if "gender" in c.strip().lower():
            gender_col = c; break
    group_col = None
    for c in df.columns:
        if c.strip().lower() == "group":
            group_col = c; break
    if group_col is None:
        raise ValueError(f"[MILOS] Cannot find 'Group' column in {path}")

    # Add normalized WBIC column for matching
    df["_WBIC_no_commas"] = df[w_col].apply(normalize_milos_wbic)
    return df, w_col, p_col, age_col, gender_col, group_col

# -------------------------- Main merge routine --------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Merge connectivity .txt + ROI labels with metadata from 3 sites into a single .npy and generate summary tables.")
    ap.add_argument("--mat-dir", default="/mnt/raid/data_cambridge/MIND_out", help="Directory containing per-subject connectivity .txt and *_regions.tsv files.")
    ap.add_argument("--jt-xlsx", default="/mnt/raid/data_cambridge/original_files/Baseline, Longitudinal & Change Combined.xlsx", help="JT site Excel file.")
    ap.add_argument("--cat-xlsx", default="/mnt/raid/data_cambridge/original_files/clinical data.xlsx", help="CAT site Excel file.")
    ap.add_argument("--milos-xlsx", default="/mnt/raid/data_cambridge/original_files/MILOS Participant information with age sex.xlsx", help="MILOS site Excel file.")
    ap.add_argument("--out-npy", default="all_subjects.npy", help="Output .npy path for merged object array.")
    ap.add_argument("--out-summary-dir", default="summary_out", help="Directory to save per-site and overall summaries (TSV).")
    return ap.parse_args()

def load_matrix_and_regions(txt_path: Path):
    """Load connectivity matrix (.txt) and regions from sibling *_regions.tsv."""
    mat = np.loadtxt(txt_path, dtype=float)
    reg_path = txt_path.with_name(txt_path.stem + "_regions.tsv")
    if not reg_path.exists():
        # Alternatively some pipelines write <subject>_regions.tsv
        alt = txt_path.with_name(txt_path.stem.replace(".txt","") + "_regions.tsv")
        if alt.exists():
            reg_path = alt
        else:
            raise FileNotFoundError(f"Regions TSV not found for {txt_path.name}")
    reg_df = pd.read_csv(reg_path, sep="\t")
    # Expected columns: index, region
    if "region" not in reg_df.columns:
        # Try variants
        col = None
        for c in reg_df.columns:
            if "region" in c.lower():
                col = c; break
        if col is None:
            raise ValueError(f"Regions TSV missing 'region' column: {reg_path}")
        regions = reg_df[col].astype(str).values
    else:
        regions = reg_df["region"].astype(str).values
    return mat, regions

def main():
    args = parse_args()
    mat_dir = Path(args.mat_dir).expanduser().resolve()
    out_summary_dir = Path(args.out_summary_dir).expanduser().resolve()
    out_summary_dir.mkdir(parents=True, exist_ok=True)

    # Load site sheets
    jt_df, jt_subj_col, jt_diag_col, jt_gender_col, jt_age_col, jt_mmse_col = load_jt_xlsx(Path(args.jt_xlsx))
    cat_df, cat_sid_col, cat_diag_col, cat_mmse_col = load_cat_xlsx(Path(args.cat_xlsx))
    mil_df, mil_wbic_col, mil_pid_col, mil_age_col, mil_gender_col, mil_group_col = load_milos_xlsx(Path(args.milos_xlsx))

    # Build quick lookup dicts
    # JT: key = normalized subject id (ends with A), row = Series
    jt_lookup = {str(row["_subj_norm"]): row for _, row in jt_df.iterrows()}

    # CAT: key = CAT NO (string)
    cat_lookup = {str(row[cat_sid_col]).strip(): row for _, row in cat_df.iterrows()}

    # MILOS: key = _WBIC_no_commas (string)
    mil_lookup = {str(row["_WBIC_no_commas"]): row for _, row in mil_df.iterrows()}

    # Iterate matrices
    entries = []
    for txt_path in sorted(mat_dir.glob("*.txt")):
        sid = txt_path.stem  # e.g., CAT007, JT_D123A, 23456
        site = detect_site(sid)

        # Load matrix/regions
        try:
            mat, regions = load_matrix_and_regions(txt_path)
        except Exception as e:
            print(f"[WARN] Skip {sid}: cannot load matrix/regions ({e})")
            continue

        age = None
        gender = None
        mmse = None
        label = None

        # Fetch metadata + map
        if site == "JT":
            lookup_id = normalize_jt_id_for_lookup(sid)
            row = jt_lookup.get(lookup_id)
            if row is None:
                print(f"[WARN][JT] No metadata for {sid} (lookup {lookup_id})")
                continue
            # Extract raw fields
            raw_diag = row[jt_diag_col] if jt_diag_col is not None and jt_diag_col in row else None
            raw_gender = row[jt_gender_col] if jt_gender_col is not None and jt_gender_col in row else None
            raw_age = row[jt_age_col] if jt_age_col is not None and jt_age_col in row else None
            raw_mmse = row[jt_mmse_col] if jt_mmse_col is not None and jt_mmse_col in row else None

            label = map_label("JT", raw_diag)
            gender = map_gender(raw_gender, "JT")
            age = float(raw_age) if pd.notna(raw_age) else None
            mmse = float(raw_mmse) if pd.notna(raw_mmse) else None

        elif site == "CAT":
            row = cat_lookup.get(sid)
            if row is None:
                print(f"[WARN][CAT] No metadata for {sid}")
                continue
            raw_diag = row[cat_diag_col]
            label = map_label("CAT", raw_diag)
            gender = None  # missing in CAT
            age = None     # missing in CAT
            if cat_mmse_col is not None and cat_mmse_col in row and pd.notna(row[cat_mmse_col]):
                try:
                    mmse = float(row[cat_mmse_col])
                except Exception:
                    mmse = None

        else:  # MILOS
            row = mil_lookup.get(sid)  # filename equals WBIC without commas
            if row is None:
                print(f"[WARN][MILOS] No metadata for {sid} using WBIC ID")
                continue
            raw_group = row[mil_group_col]
            label = map_label("MILOS", raw_group)
            # Age
            if mil_age_col is not None and mil_age_col in row and pd.notna(row[mil_age_col]):
                try:
                    age = float(row[mil_age_col])
                except Exception:
                    age = None
            # Gender
            if mil_gender_col is not None and mil_gender_col in row:
                gender = map_gender(row[mil_gender_col], "MILOS")
            # No MMSE
            mmse = None

        # Skip if label not found
        if label is None:
            print(f"[WARN] Skip {sid}: could not resolve label")
            continue

        # Basic sanity: regions length matches matrix shape
        if mat.shape[0] != mat.shape[1]:
            print(f"[WARN] Skip {sid}: matrix not square {mat.shape}")
            continue
        if len(regions) != mat.shape[0]:
            print(f"[WARN] Skip {sid}: regions length {len(regions)} != matrix size {mat.shape[0]}")
            continue

        entry = {
            "subject_id": sid,
            "site": site,
            "matrix": mat,
            "regions": regions,
            "age": age,
            "gender": gender,  # 0 male, 1 female, or None
            "mmse": mmse,
            "label": int(label)  # 0 NC, 1 MCI, 2 AD, 3 LBD
        }
        entries.append(entry)

    if not entries:
        raise RuntimeError("No valid subjects collected. Nothing to save.")

    # Save .npy (object array of dicts)
    out_npy = Path(args.out_npy).expanduser().resolve()
    np.save(out_npy, np.array(entries, dtype=object))
    print(f"[OK] Saved merged npy with {len(entries)} subjects -> {out_npy}")

    # ---------------- Stats ----------------
    # Build a compact DataFrame for stats
    stats_rows = []
    for e in entries:
        stats_rows.append({
            "site": e["site"],
            "label": e["label"],   # 0 NC, 1 MCI, 2 AD, 3 LBD
            "age": np.nan if e["age"] is None else float(e["age"]),
            "gender": e["gender"]  # 0/1 or None
        })
    sdf = pd.DataFrame(stats_rows)

    # Helper to compute summary table
    def summarize(df: pd.DataFrame, by_site: bool):
        group_fields = ["label"]
        if by_site:
            group_fields = ["site", "label"]

        # counts per group
        cnt = df.groupby(group_fields)["label"].count().rename("count").reset_index()

        # age mean/std
        age_agg = df.groupby(group_fields)["age"].agg(["mean", "std"]).reset_index().rename(columns={"mean":"age_mean","std":"age_std"})

        # gender counts: male (0), female (1)
        # We'll pivot counts of gender values, ignoring None/NaN
        g = df.dropna(subset=["gender"])
        g_cnt = g.groupby(group_fields + ["gender"]).size().rename("n").reset_index()
        # pivot gender column into two columns
        g_piv = g_cnt.pivot_table(index=group_fields, columns="gender", values="n", fill_value=0)
        # rename columns 0->male_n, 1->female_n
        g_piv = g_piv.rename(columns={0:"male_n", 1:"female_n"}).reset_index()

        # Merge all
        out = cnt.merge(age_agg, on=group_fields, how="left").merge(g_piv, on=group_fields, how="left")
        out["male_n"] = out["male_n"].fillna(0).astype(int)
        out["female_n"] = out["female_n"].fillna(0).astype(int)
        # order by label
        out = out.sort_values(group_fields).reset_index(drop=True)
        return out

    site_summary = summarize(sdf, by_site=True)
    overall_summary = summarize(sdf, by_site=False)

    # Map label codes to names for readability in TSV
    label_name = {0:"NC", 1:"MCI", 2:"AD", 3:"LBD"}
    site_summary["label_name"] = site_summary["label"].map(label_name)
    overall_summary["label_name"] = overall_summary["label"].map(label_name)

    # Save summaries
    site_summary = site_summary[["site","label","label_name","count","age_mean","age_std","male_n","female_n"]]
    overall_summary = overall_summary[["label","label_name","count","age_mean","age_std","male_n","female_n"]]

    site_path = out_summary_dir / "summary_by_site.tsv"
    overall_path = out_summary_dir / "summary_overall.tsv"
    site_summary.to_csv(site_path, sep="\t", index=False)
    overall_summary.to_csv(overall_path, sep="\t", index=False)

    print(f"[OK] Saved per-site summary -> {site_path}")
    print(f"[OK] Saved overall summary  -> {overall_path}")


if __name__ == "__main__":
    main()
