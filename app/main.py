import logging
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query

from app.model import DEFAULT_MODEL, ModelType, get_categories, load_model, predict
from app.schemas import PredictRequest, PredictResponse

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

VALID_MODELS: list[str] = ["tfidf_lr", "tfidf_nb", "transformer"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load the default model (tfidf_lr) on startup."""
    load_model(DEFAULT_MODEL)
    yield


app = FastAPI(
    title="Invoice Expense Classifier",
    description=(
        "Classifies invoice text into expense categories. "
        "Supports three models: **tfidf_lr** (default), **tfidf_nb**, **transformer**."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.post("/predict", response_model=PredictResponse, summary="Predict expense category")
def predict_category(
    request: PredictRequest,
    model: Annotated[
        str,
        Query(description="Model backend: tfidf_lr | tfidf_nb | transformer"),
    ] = DEFAULT_MODEL,
):
    """
    Accepts invoice text and returns the predicted expense category
    with a confidence score (0–1).

    **model** query param selects the backend:
    - `tfidf_lr` – TF-IDF + Logistic Regression *(default, fastest)*
    - `tfidf_nb` – TF-IDF + Naive Bayes (ComplementNB)
    - `transformer` – Sentence Transformer (all-MiniLM-L6-v2) + LR *(first call downloads ~80 MB)*
    """
    if model not in VALID_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model '{model}'. Choose from: {VALID_MODELS}",
        )
    try:
        result = predict(request.text, model_type=model)  # type: ignore[arg-type]
        return result
    except Exception as exc:
        logger.exception("Prediction failed: %s", exc)
        raise HTTPException(status_code=500, detail="Prediction failed. Please try again.")


@app.get("/categories", summary="List supported categories")
def list_categories(
    model: Annotated[str, Query(description="Model backend")] = DEFAULT_MODEL,
):
    """Returns all supported expense categories for the chosen model."""
    if model not in VALID_MODELS:
        raise HTTPException(status_code=400, detail=f"Invalid model '{model}'.")
    return {"categories": get_categories(model_type=model)}  # type: ignore[arg-type]


@app.get("/health", summary="Health check")
def health_check():
    """Returns API health status."""
    return {"status": "ok"}
