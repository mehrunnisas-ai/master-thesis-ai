#!/usr/bin/env python3
"""
Subgroup fairness audit for student-dropout prediction
======================================================
Dataset : "Predict Students' Dropout and Academic Success" (Realinho et al., 2021),
          the same 4,424-student dataset used in the MSc dissertation.
Usage   : python fairness_analysis.py --csv data.csv
          (or run without --csv inside Colab with internet: it will try the UCI mirror via ucimlrepo)

What it does
------------
1. Builds a binary task: Dropout (1) vs. Graduate/Enrolled (0).
2. Trains Logistic Regression and Random Forest with 5-fold stratified cross-validation,
   so every student gets an out-of-fold prediction (no student is scored by a model that saw them).
3. Reports, per protected group: n, observed dropout rate, predicted-dropout (selection) rate,
   accuracy, true-positive rate (TPR), false-positive rate (FPR), precision.
4. Reports between-group gaps with bootstrap 95% confidence intervals.
5. Tries two simple mitigations on the chosen attribute and compares them with the baseline:
   (a) reweighing (Kamiran & Calders, 2012)   (b) group-specific decision thresholds that equalise TPR.
6. Runs everything for two feature sets: enrolment-only and enrolment + first-year performance.

Outputs (written to --out, default ./fairness_output):
   group_metrics.csv, gaps.csv, mitigation.csv, tpr_by_group.png

No result is hard-coded: every number in the paper must come from these files.
"""
import argparse
import os
import re
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RNG = 42


# --------------------------------------------------------------------------- data
def norm(c: str) -> str:
    """Lower-case, strip punctuation/spaces so 'Age at enrolment' == 'age_at_enrollment'."""
    c = c.strip().lower().replace("enrolment", "enrollment").replace("nacionality", "nationality")
    return re.sub(r"[^a-z0-9]+", "_", c).strip("_")


def load(csv):
    if csv:
        df = pd.read_csv(csv, sep=None, engine="python")  # auto-detects ; or ,
    else:
        from ucimlrepo import fetch_ucirepo  # pip install ucimlrepo
        ds = fetch_ucirepo(id=697)
        df = pd.concat([ds.data.features, ds.data.targets], axis=1)
    df.columns = [norm(c) for c in df.columns]
    return df


def prepare(df):
    target_col = "target"
    y = (df[target_col].astype(str).str.lower() == "dropout").astype(int)

    g = {}
    g["gender"] = df["gender"].map({0: "female", 1: "male"})
    age = df["age_at_enrollment"]
    g["age_group"] = pd.cut(age, [0, 20, 25, 200], labels=["<=20", "21-25", ">25"]).astype(str)
    nat_col = "nationality" if "nationality" in df.columns else None
    if nat_col:
        # In the source data, nationality code 1 = Portuguese.
        g["nationality"] = np.where(df[nat_col] == 1, "Portuguese", "other")
    g["scholarship"] = df["scholarship_holder"].map({0: "no", 1: "yes"})
    if "displaced" in df.columns:
        g["displaced"] = df["displaced"].map({0: "no", 1: "yes"})
    groups = pd.DataFrame(g)

    perf_cols = [c for c in df.columns if c.startswith("curricular_units_1st_sem")]
    perf_cols += [c for c in df.columns if c.startswith("curricular_units_2nd_sem")]
    drop = set([target_col] + perf_cols)
    enrol_cols = [c for c in df.columns if c not in drop]
    feature_sets = {
        "enrolment_only": enrol_cols,
        "enrolment_plus_year1": enrol_cols + [c for c in perf_cols if "2nd" not in c],
    }
    return y, groups, feature_sets


# ------------------------------------------------------------------------ metrics
def rates(y, p, s=None):
    y, p = np.asarray(y), np.asarray(p)
    tp = ((p == 1) & (y == 1)).sum()
    fp = ((p == 1) & (y == 0)).sum()
    tn = ((p == 0) & (y == 0)).sum()
    fn = ((p == 0) & (y == 1)).sum()
    d = lambda a, b: a / b if b else np.nan
    return dict(
        n=len(y),
        observed_dropout=y.mean(),
        selection_rate=p.mean(),
        accuracy=d(tp + tn, len(y)),
        tpr=d(tp, tp + fn),
        fpr=d(fp, fp + tn),
        precision=d(tp, tp + fp),
    )


def group_table(y, p, grp, model, fset, attr):
    rows = []
    grp = np.asarray(grp)
    for level in sorted(pd.unique(grp)):
        m = np.asarray(grp == level)
        if m.sum() == 0:
            continue
        rows.append(dict(model=model, feature_set=fset, attribute=attr, group=level, **rates(y[m], p[m])))
    return pd.DataFrame(rows)


def gaps(y, p, grp, n_boot=500, min_n=30):
    """Max-min gap across groups with >= min_n students; bootstrap CI."""
    y, p, grp = np.asarray(y), np.asarray(p), np.asarray(grp)
    levels = [l for l in np.unique(grp) if (grp == l).sum() >= min_n]

    def one(idx):
        per = {l: rates(y[idx][grp[idx] == l], p[idx][grp[idx] == l]) for l in levels}
        sel = [v["selection_rate"] for v in per.values()]
        tpr = [v["tpr"] for v in per.values()]
        fpr = [v["fpr"] for v in per.values()]
        acc = [v["accuracy"] for v in per.values()]
        rng = lambda a: np.nanmax(a) - np.nanmin(a)
        return dict(
            demographic_parity_gap=rng(sel),
            equal_opportunity_gap=rng(tpr),
            equalized_odds_gap=max(rng(tpr), rng(fpr)),
            accuracy_gap=rng(acc),
        )

    point = one(np.arange(len(y)))
    rs = np.random.RandomState(RNG)
    boots = [one(rs.randint(0, len(y), len(y))) for _ in range(n_boot)]
    out = {}
    for k, v in point.items():
        b = np.array([x[k] for x in boots])
        out[k] = v
        out[k + "_ci_low"], out[k + "_ci_high"] = np.nanpercentile(b, [2.5, 97.5])
    return out


# ---------------------------------------------------------------------- mitigation
def make_model(name):
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
    return RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced_subsample",
                                  random_state=RNG, n_jobs=-1)


def reweigh(y, grp):
    """Kamiran & Calders reweighing: w(g,y) = P(g)P(y) / P(g,y)."""
    y, grp = np.asarray(y), np.asarray(grp)
    w = np.ones(len(y))
    for g in np.unique(grp):
        for c in (0, 1):
            m = (grp == g) & (y == c)
            if m.sum():
                w[m] = ((grp == g).mean() * (y == c).mean()) / m.mean()
    return w


def oof_scores(X, y, name, weights=None):
    skf = StratifiedKFold(5, shuffle=True, random_state=RNG)
    s = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        m = make_model(name)
        fit_kw = {}
        if weights is not None:
            key = "logisticregression__sample_weight" if name == "logreg" else "sample_weight"
            fit_kw[key] = weights[tr]
        m.fit(X.iloc[tr], y[tr], **fit_kw)
        s[te] = m.predict_proba(X.iloc[te])[:, 1]
    return s


def group_thresholds(X, y, grp, name):
    """Nested: thresholds are learned on the TRAINING part of each fold only, then applied to the test part."""
    skf = StratifiedKFold(5, shuffle=True, random_state=RNG)
    pred = np.zeros(len(y), dtype=int)
    grp = np.asarray(grp)
    for tr, te in skf.split(X, y):
        inner = cross_val_predict(make_model(name), X.iloc[tr], y[tr], cv=3, method="predict_proba")[:, 1]
        pos = y[tr] == 1
        # overall TPR at 0.5 on the training part is the target for every group
        target_tpr = (inner[pos] >= 0.5).mean()
        thr = {}
        for g in np.unique(grp[tr]):
            m = (grp[tr] == g) & pos
            thr[g] = np.quantile(inner[m], 1 - target_tpr) if m.sum() >= 10 else 0.5
        final = make_model(name).fit(X.iloc[tr], y[tr])
        s = final.predict_proba(X.iloc[te])[:, 1]
        pred[te] = [int(si >= thr.get(g, 0.5)) for si, g in zip(s, grp[te])]
    return pred


# -------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--out", default="fairness_output")
    ap.add_argument("--mitigate_on", default="gender")
    ap.add_argument("--boot", type=int, default=500)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    df = load(a.csv)
    y, groups, fsets = prepare(df)
    y = y.to_numpy()
    print(f"n = {len(y)}, observed dropout rate = {y.mean():.3f}")

    gm, gp, mit = [], [], []
    for fset, cols in fsets.items():
        X = df[cols].apply(pd.to_numeric, errors="coerce").fillna(0)
        for name in ("logreg", "rf"):
            s = oof_scores(X, y, name)
            p = (s >= 0.5).astype(int)
            overall = rates(y, p)
            print(f"[{fset}/{name}] overall accuracy={overall['accuracy']:.3f} TPR={overall['tpr']:.3f}")
            for attr in groups.columns:
                grp = groups[attr].to_numpy()
                gm.append(group_table(y, p, grp, name, fset, attr))
                gp.append(dict(model=name, feature_set=fset, attribute=attr, **gaps(y, p, grp, a.boot)))

            # ---- mitigation on one attribute
            att = a.mitigate_on
            grp = groups[att].to_numpy()
            variants = {
                "baseline": p,
                "reweighing": (oof_scores(X, y, name, reweigh(y, grp)) >= 0.5).astype(int),
                "group_thresholds": group_thresholds(X, y, grp, name),
            }
            for vname, vp in variants.items():
                r = rates(y, vp)
                g = gaps(y, vp, grp, a.boot)
                mit.append(dict(model=name, feature_set=fset, attribute=att, method=vname,
                                accuracy=r["accuracy"], tpr=r["tpr"], fpr=r["fpr"], **g))

    pd.concat(gm).to_csv(f"{a.out}/group_metrics.csv", index=False)
    pd.DataFrame(gp).to_csv(f"{a.out}/gaps.csv", index=False)
    pd.DataFrame(mit).to_csv(f"{a.out}/mitigation.csv", index=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        t = pd.concat(gm)
        t = t[(t.model == "logreg") & (t.feature_set == "enrolment_only") & (t.n >= 30)]
        t["label"] = t.attribute + ": " + t.group.astype(str)
        t = t.sort_values(["attribute", "group"])
        plt.figure(figsize=(8, 5))
        plt.barh(t["label"], t["tpr"])
        plt.xlabel("True-positive rate for dropout (higher = more dropouts caught)")
        plt.title("Logistic regression, enrolment-only features")
        plt.tight_layout()
        plt.savefig(f"{a.out}/tpr_by_group.png", dpi=200)
    except Exception as e:  # plotting is optional
        print("plot skipped:", e)
    print("done ->", a.out)


if __name__ == "__main__":
    sys.exit(main())
