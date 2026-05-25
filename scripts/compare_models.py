"""
compare_models.py
─────────────────────────────────────────────────────────────────────────────
Trains and evaluates three classifiers on the invoice expense dataset:

  Model 1 – TF-IDF + Logistic Regression
  Model 2 – TF-IDF + Naive Bayes (ComplementNB)
  Model 3 – Sentence Transformer (all-MiniLM-L6-v2) + Logistic Regression

Results are printed to stdout AND saved to:
  results/model_comparison.txt

Usage:
  python scripts/compare_models.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline

BASE_DIR    = Path(__file__).resolve().parent.parent
DATA_PATH   = BASE_DIR / "data" / "training_data.csv"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_PATH = RESULTS_DIR / "model_comparison.txt"

CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
SEP  = "=" * 70
SEP2 = "-" * 70


# ── Model builders ─────────────────────────────────────────────────────────────

def build_tfidf_lr() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2), max_features=10_000,
            sublinear_tf=True, stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000, C=5.0, solver="lbfgs", random_state=42)),
    ])


def build_tfidf_nb() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2), max_features=10_000,
            sublinear_tf=False, stop_words="english")),
        ("clf", ComplementNB(alpha=0.1)),
    ])


# ── Evaluation helpers ─────────────────────────────────────────────────────────

def _format_section(
    name: str,
    cv_scores: np.ndarray,
    cv_time: float,
    test_acc: float,
    n_test: int,
    train_ms: float,
    infer_ms: float,
    report: str,
    conf_matrix: np.ndarray,
    labels: list,
    extra_lines: list[str] | None = None,
) -> tuple[str, float, float]:
    lines = [
        f"\n{SEP}",
        f"  MODEL : {name}",
        SEP,
    ]
    if extra_lines:
        lines.extend(extra_lines)
    lines += [
        f"  CV Accuracy (5-fold)  : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}",
        f"  Per-fold scores       : {[f'{s:.4f}' for s in cv_scores]}",
        f"  CV wall-clock time    : {cv_time:.2f}s",
        f"  Test Accuracy         : {test_acc:.4f}  "
            f"({int(round(test_acc * n_test))}/{n_test} correct)",
        f"  Train time            : {train_ms:.1f} ms",
        f"  Inference time        : {infer_ms:.3f} ms  "
            f"({n_test} samples, {infer_ms/n_test*1000:.2f} µs/sample)",
        f"\n  Classification Report:",
        SEP2,
    ]
    for l in report.split("\n"):
        lines.append(f"    {l}")
    lines += [
        SEP2,
        f"\n  Confusion Matrix  (rows=actual, cols=predicted):",
        f"  Labels: {labels}",
    ]
    for row in conf_matrix:
        lines.append(f"    {list(row)}")
    return "\n".join(lines), float(cv_scores.mean()), float(test_acc)


def evaluate_sklearn(name, model_fn, X, y, X_train, X_test, y_train, y_test, labels):
    model = model_fn()
    t0 = time.perf_counter()
    cv_scores = cross_val_score(model, X, y, cv=CV, scoring="accuracy")
    cv_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    infer_ms = (time.perf_counter() - t0) * 1000

    test_acc = accuracy_score(y_test, y_pred)
    report   = classification_report(y_test, y_pred, digits=4)
    cm       = confusion_matrix(y_test, y_pred, labels=labels)

    section, cv_acc, t_acc = _format_section(
        name, cv_scores, cv_time, test_acc, len(y_test),
        train_ms, infer_ms, report, cm, labels,
    )
    return section, cv_acc, t_acc, train_ms, infer_ms


def evaluate_transformer(X, y, X_train, X_test, y_train, y_test, labels):
    name = "Sentence Transformer (all-MiniLM-L6-v2) + Logistic Regression"
    print(f"\n  [Transformer] Loading model...", flush=True)
    from sentence_transformers import SentenceTransformer
    encoder = SentenceTransformer("all-MiniLM-L6-v2")

    print("  [Transformer] Encoding all text...", flush=True)
    t0 = time.perf_counter()
    X_enc       = encoder.encode(X.tolist(),       show_progress_bar=False, batch_size=64)
    X_train_enc = encoder.encode(X_train.tolist(), show_progress_bar=False, batch_size=64)
    X_test_enc  = encoder.encode(X_test.tolist(),  show_progress_bar=False, batch_size=64)
    enc_time = time.perf_counter() - t0
    print(f"  [Transformer] Encoded {len(X)} samples in {enc_time:.2f}s", flush=True)

    clf = LogisticRegression(max_iter=1000, C=5.0, random_state=42)

    t0 = time.perf_counter()
    cv_scores = cross_val_score(clf, X_enc, y, cv=CV, scoring="accuracy")
    cv_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    clf.fit(X_train_enc, y_train)
    train_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    y_pred = clf.predict(X_test_enc)
    infer_ms_lr = (time.perf_counter() - t0) * 1000

    # Realistic per-sample latency (encode + predict, averaged over 20 samples)
    sample_texts = X_test.tolist()[:20]
    t0 = time.perf_counter()
    for txt in sample_texts:
        e = encoder.encode([txt], show_progress_bar=False)
        clf.predict(e)
    per_sample_ms = (time.perf_counter() - t0) / len(sample_texts) * 1000

    test_acc = accuracy_score(y_test, y_pred)
    report   = classification_report(y_test, y_pred, digits=4)
    cm       = confusion_matrix(y_test, y_pred, labels=labels)

    extra = [
        f"  Encoding time (full dataset) : {enc_time:.2f}s  ({len(X)} samples)",
        f"  Realistic per-sample latency : ~{per_sample_ms:.1f} ms  (encode + LR predict)",
    ]
    section, cv_acc, t_acc = _format_section(
        name, cv_scores, cv_time, test_acc, len(y_test),
        train_ms, infer_ms_lr, report, cm, labels, extra_lines=extra,
    )
    return section, cv_acc, t_acc, train_ms, infer_ms_lr, per_sample_ms


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    print("Loading data...", flush=True)
    df = pd.read_csv(DATA_PATH)
    X, y = df["text"], df["category"]
    labels = sorted(y.unique())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    timestamp = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")
    header = "\n".join([
        SEP,
        "  INVOICE EXPENSE CLASSIFIER — MODEL COMPARISON REPORT",
        f"  Generated : {timestamp}",
        f"  Command   : python scripts/compare_models.py",
        SEP,
        f"  Dataset       : {len(X)} total samples  "
            f"({len(X_train)} train / {len(X_test)} test, 80/20 stratified split)",
        f"  Categories    : {labels}",
        f"  CV Strategy   : StratifiedKFold, 5 folds, random_state=42",
        f"  Test split    : random_state=42, stratify=y",
        SEP,
    ])

    output_parts = [header]
    summary_rows: list[dict] = []

    # ── Model 1: TF-IDF + LR ──────────────────────────────────────────────────
    print("\nEvaluating Model 1: TF-IDF + Logistic Regression...", flush=True)
    sec, cv_acc, t_acc, tr_ms, inf_ms = evaluate_sklearn(
        "TF-IDF + Logistic Regression",
        build_tfidf_lr, X, y, X_train, X_test, y_train, y_test, labels,
    )
    output_parts.append(sec)
    summary_rows.append({"model": "TF-IDF + Logistic Regression",
                          "cv_acc": cv_acc, "test_acc": t_acc,
                          "train_ms": tr_ms, "infer_ms": inf_ms})

    # ── Model 2: TF-IDF + Naive Bayes ─────────────────────────────────────────
    print("\nEvaluating Model 2: TF-IDF + Naive Bayes (ComplementNB)...", flush=True)
    sec, cv_acc, t_acc, tr_ms, inf_ms = evaluate_sklearn(
        "TF-IDF + Naive Bayes (ComplementNB)",
        build_tfidf_nb, X, y, X_train, X_test, y_train, y_test, labels,
    )
    output_parts.append(sec)
    summary_rows.append({"model": "TF-IDF + Naive Bayes (ComplementNB)",
                          "cv_acc": cv_acc, "test_acc": t_acc,
                          "train_ms": tr_ms, "infer_ms": inf_ms})

    # ── Model 3: Sentence Transformer + LR ────────────────────────────────────
    print("\nEvaluating Model 3: Sentence Transformer + LR...", flush=True)
    sec, cv_acc, t_acc, tr_ms, inf_ms, per_ms = evaluate_transformer(
        X, y, X_train, X_test, y_train, y_test, labels
    )
    output_parts.append(sec)
    summary_rows.append({"model": "Sentence Transformer (all-MiniLM-L6-v2) + LR",
                          "cv_acc": cv_acc, "test_acc": t_acc,
                          "train_ms": tr_ms, "infer_ms": inf_ms,
                          "per_sample_ms": per_ms})

    # ── Summary table ─────────────────────────────────────────────────────────
    best_cv   = max(r["cv_acc"]   for r in summary_rows)
    best_test = max(r["test_acc"] for r in summary_rows)

    table = [
        f"\n{SEP}",
        "  SUMMARY COMPARISON TABLE",
        SEP,
        f"  {'Model':<50} {'CV Acc':>8}  {'Test Acc':>9}  {'Train(ms)':>10}  {'Infer(ms)':>10}",
        f"  {SEP2}",
    ]
    for r in summary_rows:
        cv_flag   = " ◄ BEST" if r["cv_acc"]   == best_cv   else ""
        test_flag = " ◄ BEST" if r["test_acc"] == best_test else ""
        ps = f"  (~{r['per_sample_ms']:.1f}ms/sample)" if "per_sample_ms" in r else ""
        table.append(
            f"  {r['model']:<50} {r['cv_acc']:>8.4f}{cv_flag}"
        )
        table.append(
            f"  {'':50} {'':>8}  {r['test_acc']:>9.4f}{test_flag}  "
            f"{r['train_ms']:>10.1f}  {r['infer_ms']:>10.3f}{ps}"
        )
        table.append(f"  {SEP2}")

    table += [
        f"  ◄ BEST = highest value in that column",
        SEP,
    ]

    # ── Recommendation ─────────────────────────────────────────────────────────
    best_overall = max(summary_rows, key=lambda r: r["cv_acc"])
    rec = [
        f"\n  RECOMMENDATION",
        SEP2,
        f"  Best CV accuracy  : '{best_overall['model']}' → {best_overall['cv_acc']:.4f}",
        f"",
        f"  For production (fast inference) : use tfidf_lr  (POST /predict?model=tfidf_lr)",
        f"  For best accuracy               : use {best_overall['model'].split()[0].lower()}_"
          f"{'lr' if 'Logistic' in best_overall['model'] else 'nb' if 'Bayes' in best_overall['model'] else 'transformer'}",
        f"  For semantic understanding      : use transformer  (POST /predict?model=transformer)",
        SEP,
    ]

    output_parts.extend(table)
    output_parts.extend(rec)

    full_output = "\n".join(output_parts)
    print(full_output, flush=True)

    RESULTS_DIR.mkdir(exist_ok=True)
    RESULTS_PATH.write_text(full_output, encoding="utf-8")
    print(f"\n✓  Results saved → {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
