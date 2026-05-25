"""
train.py – Standalone training script with evaluation metrics.

Usage:
    python scripts/train.py                          # train all 3 models
    python scripts/train.py --model-type tfidf_lr    # train one specific model
    python scripts/train.py --data data/training_data.csv --model-type tfidf_nb
"""

import argparse
import logging
import time
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_DATA  = BASE_DIR / "data" / "training_data.csv"
MODEL_PATHS = {
    "tfidf_lr":    BASE_DIR / "models" / "tfidf_lr.joblib",
    "tfidf_nb":    BASE_DIR / "models" / "tfidf_nb.joblib",
    "transformer": BASE_DIR / "models" / "transformer.joblib",
}


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


# ── Per-model training ─────────────────────────────────────────────────────────

def train_model(
    model_type: str,
    data_path: Path,
    model_path: Path,
    test_size: float = 0.2,
    cv_folds: int = 5,
):
    logger.info("─" * 60)
    logger.info("Training: %s", model_type)
    logger.info("─" * 60)

    df = pd.read_csv(data_path)
    if "text" not in df.columns or "category" not in df.columns:
        raise ValueError("CSV must contain 'text' and 'category' columns.")

    logger.info("Dataset: %d rows | Categories: %s", len(df), sorted(df["category"].unique()))
    X, y = df["text"], df["category"]
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    if model_type in ("tfidf_lr", "tfidf_nb"):
        pipeline = build_tfidf_lr() if model_type == "tfidf_lr" else build_tfidf_nb()

        # Cross-validation
        logger.info("Running %d-fold cross-validation...", cv_folds)
        t0 = time.perf_counter()
        cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
        logger.info(
            "CV Accuracy: %.4f ± %.4f  (folds: %s)  [%.2fs]",
            cv_scores.mean(), cv_scores.std(),
            [f"{s:.4f}" for s in cv_scores],
            time.perf_counter() - t0,
        )

        # Train/test eval
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y)
        eval_pipeline = build_tfidf_lr() if model_type == "tfidf_lr" else build_tfidf_nb()
        eval_pipeline.fit(X_train, y_train)
        logger.info("\n%s", classification_report(y_test, eval_pipeline.predict(X_test)))

        # Final fit on all data
        pipeline.fit(X, y)
        final_pipeline = pipeline

    elif model_type == "transformer":
        from sentence_transformers import SentenceTransformer

        logger.info("Loading SentenceTransformer (all-MiniLM-L6-v2)...")
        encoder = SentenceTransformer("all-MiniLM-L6-v2")

        logger.info("Encoding dataset...")
        t0 = time.perf_counter()
        X_enc = encoder.encode(X.tolist(), show_progress_bar=False, batch_size=64)
        logger.info("Encoded %d samples in %.2fs", len(X), time.perf_counter() - t0)

        clf = LogisticRegression(max_iter=1000, C=5.0, random_state=42)

        logger.info("Running %d-fold cross-validation on embeddings...", cv_folds)
        cv_scores = cross_val_score(clf, X_enc, y, cv=cv, scoring="accuracy")
        logger.info(
            "CV Accuracy: %.4f ± %.4f  (folds: %s)",
            cv_scores.mean(), cv_scores.std(),
            [f"{s:.4f}" for s in cv_scores],
        )

        X_train, X_test, y_train, y_test = train_test_split(
            X_enc, y, test_size=test_size, random_state=42, stratify=y)
        eval_clf = LogisticRegression(max_iter=1000, C=5.0, random_state=42)
        eval_clf.fit(X_train, y_train)
        logger.info("\n%s", classification_report(y_test, eval_clf.predict(X_test)))

        # Build saveable pipeline wrapper
        # Re-use TransformerPipeline from app.model
        import sys
        sys.path.insert(0, str(BASE_DIR))
        from app.model import TransformerPipeline
        final_pipeline = TransformerPipeline()
        final_pipeline.fit(X, y)
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipeline, model_path)
    logger.info("Saved → %s", model_path)
    return final_pipeline


def train_and_save(
    data_path: Path = DEFAULT_DATA,
    model_type: str = "all",
    test_size: float = 0.2,
    cv_folds: int = 5,
):
    types = list(MODEL_PATHS.keys()) if model_type == "all" else [model_type]
    for mt in types:
        train_model(mt, data_path, MODEL_PATHS[mt], test_size, cv_folds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train invoice classifier model(s).")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="Path to training CSV")
    parser.add_argument(
        "--model-type",
        choices=["all", "tfidf_lr", "tfidf_nb", "transformer"],
        default="all",
        help="Which model to train (default: all)",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--cv-folds", type=int, default=5)
    args = parser.parse_args()

    train_and_save(
        data_path=args.data,
        model_type=args.model_type,
        test_size=args.test_size,
        cv_folds=args.cv_folds,
    )

