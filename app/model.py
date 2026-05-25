import logging
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "training_data.csv"

ModelType = Literal["tfidf_lr", "tfidf_nb", "transformer"]
DEFAULT_MODEL: ModelType = "tfidf_lr"

MODEL_PATHS: dict[str, Path] = {
    "tfidf_lr":    BASE_DIR / "models" / "tfidf_lr.joblib",
    "tfidf_nb":    BASE_DIR / "models" / "tfidf_nb.joblib",
    "transformer": BASE_DIR / "models" / "transformer.joblib",
}

# In-memory cache – keyed by model_type
_pipelines: dict[str, object] = {}


# ── Transformer pipeline wrapper ──────────────────────────────────────────────

class TransformerPipeline:
    """
    Sentence Transformer (all-MiniLM-L6-v2) encoder + Logistic Regression.

    The SentenceTransformer weights are NOT stored in the joblib file;
    they are reloaded lazily from the local HuggingFace cache on first use.
    Only the fitted LogisticRegression (small) is persisted.
    """

    MODEL_NAME = "all-MiniLM-L6-v2"

    def __init__(self) -> None:
        self._encoder = None
        self._clf: LogisticRegression | None = None
        self.classes_ = None

    # Exclude the encoder from joblib serialisation
    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_encoder"] = None
        return state

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer  # lazy import
            logger.info("Loading SentenceTransformer '%s'...", self.MODEL_NAME)
            self._encoder = SentenceTransformer(self.MODEL_NAME)
        return self._encoder

    def _encode(self, X) -> np.ndarray:
        return self._get_encoder().encode(list(X), show_progress_bar=False, batch_size=64)

    def fit(self, X, y):
        embeddings = self._encode(X)
        self._clf = LogisticRegression(max_iter=1000, C=5.0, random_state=42)
        self._clf.fit(embeddings, y)
        self.classes_ = self._clf.classes_
        return self

    def predict(self, X):
        return self._clf.predict(self._encode(X))

    def predict_proba(self, X):
        return self._clf.predict_proba(self._encode(X))


# ── Model builders ─────────────────────────────────────────────────────────────

def _build_tfidf_lr() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2), max_features=10_000,
            sublinear_tf=True, stop_words="english")),
        ("clf", LogisticRegression(max_iter=1000, C=5.0, solver="lbfgs", random_state=42)),
    ])


def _build_tfidf_nb() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2), max_features=10_000,
            sublinear_tf=False, stop_words="english")),
        ("clf", ComplementNB(alpha=0.1)),
    ])


# ── Training & persistence ─────────────────────────────────────────────────────

def _train_and_save(model_type: ModelType, data_path: Path, model_path: Path):
    df = pd.read_csv(data_path)
    X, y = df["text"], df["category"]
    logger.info("Auto-training model '%s' from %s ...", model_type, data_path)

    if model_type == "tfidf_lr":
        pipeline = _build_tfidf_lr()
        pipeline.fit(X, y)
    elif model_type == "tfidf_nb":
        pipeline = _build_tfidf_nb()
        pipeline.fit(X, y)
    elif model_type == "transformer":
        pipeline = TransformerPipeline()
        pipeline.fit(X, y)
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    logger.info("Model '%s' saved → %s", model_type, model_path)
    return pipeline


# ── Public API ─────────────────────────────────────────────────────────────────

def load_model(model_type: ModelType = DEFAULT_MODEL) -> None:
    """Load (or auto-train) a model into the in-memory cache."""
    if model_type in _pipelines:
        return
    path = MODEL_PATHS[model_type]
    if path.exists():
        logger.info("Loading model '%s' from %s", model_type, path)
        pipeline = joblib.load(path)
    else:
        logger.warning("Model '%s' not found – auto-training from default data.", model_type)
        pipeline = _train_and_save(model_type, DATA_PATH, path)
    _pipelines[model_type] = pipeline
    logger.info("Model '%s' ready. Classes: %s", model_type, list(pipeline.classes_))


def predict(text: str, model_type: ModelType = DEFAULT_MODEL) -> dict:
    """Return predicted category and confidence score for *text*."""
    if model_type not in _pipelines:
        load_model(model_type)
    pipeline = _pipelines[model_type]
    probabilities = pipeline.predict_proba([text])[0]
    top_idx = int(probabilities.argmax())
    category: str = pipeline.classes_[top_idx]
    confidence: float = float(probabilities[top_idx])
    return {"category": category, "confidence": round(confidence, 4)}


def get_categories(model_type: ModelType = DEFAULT_MODEL) -> list[str]:
    """Return sorted list of supported expense categories."""
    if model_type not in _pipelines:
        load_model(model_type)
    return sorted(_pipelines[model_type].classes_.tolist())
