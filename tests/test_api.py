"""
Unit tests for the Invoice Expense Classification API.

Run with:
    pytest tests/ -v
Save output:
    pytest tests/ -v 2>&1 | Tee-Object results/test_results.txt   (PowerShell)
    pytest tests/ -v 2>&1 | tee results/test_results.txt          (bash)
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


# ── /health ───────────────────────────────────────────────────────────────────

class TestHealthEndpoint:
    def test_health_returns_ok(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


# ── /categories ───────────────────────────────────────────────────────────────

class TestCategoriesEndpoint:
    def test_returns_list(self):
        response = client.get("/categories")
        assert response.status_code == 200
        data = response.json()
        assert "categories" in data
        assert isinstance(data["categories"], list)

    def test_contains_all_required_categories(self):
        response = client.get("/categories")
        categories = set(response.json()["categories"])
        required = {"Logistics", "Office Supplies", "Cloud/Software", "Utilities", "Travel", "Inventory"}
        assert required.issubset(categories), f"Missing categories: {required - categories}"


# ── /predict – response structure ─────────────────────────────────────────────

class TestPredictResponseStructure:
    def test_response_has_category_and_confidence(self):
        r = client.post("/predict", json={"text": "Blue Dart courier charges for warehouse delivery"})
        assert r.status_code == 200
        assert "category" in r.json()
        assert "confidence" in r.json()

    def test_confidence_is_between_0_and_1(self):
        r = client.post("/predict", json={"text": "AWS monthly cloud hosting bill"})
        assert 0.0 <= r.json()["confidence"] <= 1.0

    def test_confidence_is_float(self):
        r = client.post("/predict", json={"text": "Electricity bill for office premises"})
        assert isinstance(r.json()["confidence"], float)


# ── /predict – default model (tfidf_lr) predictions ──────────────────────────

class TestPredictDefaultModel:
    def _p(self, text: str) -> dict:
        r = client.post("/predict", json={"text": text})
        assert r.status_code == 200
        return r.json()

    # Logistics
    def test_logistics_courier(self):
        assert self._p("Blue Dart courier charges for warehouse delivery")["category"] == "Logistics"

    def test_logistics_freight(self):
        assert self._p("Freight forwarding charges for bulk international shipment")["category"] == "Logistics"

    def test_logistics_shipping(self):
        assert self._p("DHL Express overnight shipping fees")["category"] == "Logistics"

    # Cloud/Software
    def test_cloud_aws(self):
        assert self._p("AWS monthly cloud hosting bill")["category"] == "Cloud/Software"

    def test_cloud_azure(self):
        assert self._p("Microsoft Azure compute subscription charges")["category"] == "Cloud/Software"

    def test_cloud_saas(self):
        assert self._p("Slack Pro plan monthly subscription")["category"] == "Cloud/Software"

    # Utilities
    def test_utilities_electricity(self):
        assert self._p("Electricity bill for office premises March")["category"] == "Utilities"

    def test_utilities_internet(self):
        assert self._p("Airtel broadband internet service monthly charges")["category"] == "Utilities"

    def test_utilities_mobile(self):
        assert self._p("Corporate mobile phone plan monthly invoice")["category"] == "Utilities"

    # Travel
    def test_travel_flight(self):
        assert self._p("Air India flight tickets Mumbai to Delhi")["category"] == "Travel"

    def test_travel_hotel(self):
        assert self._p("Hotel accommodation Bangalore 3 nights business trip")["category"] == "Travel"

    def test_travel_cab(self):
        assert self._p("Uber cab charges for client meeting")["category"] == "Travel"

    # Office Supplies
    def test_office_supplies_paper(self):
        assert self._p("A4 printer paper purchase 10 reams")["category"] == "Office Supplies"

    def test_office_supplies_stationery(self):
        assert self._p("Office stationery kit pens pencils and markers")["category"] == "Office Supplies"

    def test_office_supplies_toner(self):
        assert self._p("HP printer toner cartridge replacement")["category"] == "Office Supplies"

    # Inventory
    def test_inventory_raw_materials(self):
        assert self._p("Steel sheets raw material purchase for factory")["category"] == "Inventory"

    def test_inventory_spare_parts(self):
        assert self._p("Spare parts for machinery preventive maintenance")["category"] == "Inventory"

    def test_inventory_stock(self):
        assert self._p("Product stock replenishment order from vendor")["category"] == "Inventory"


# ── /predict – model query parameter ─────────────────────────────────────────

class TestModelSelection:
    def test_explicit_tfidf_lr_model(self):
        r = client.post("/predict?model=tfidf_lr", json={"text": "AWS monthly cloud hosting bill"})
        assert r.status_code == 200
        assert r.json()["category"] == "Cloud/Software"

    def test_tfidf_nb_logistics(self):
        r = client.post("/predict?model=tfidf_nb",
                        json={"text": "Blue Dart courier charges for warehouse delivery"})
        assert r.status_code == 200
        assert r.json()["category"] == "Logistics"

    def test_tfidf_nb_cloud(self):
        r = client.post("/predict?model=tfidf_nb",
                        json={"text": "AWS monthly cloud hosting bill"})
        assert r.status_code == 200
        assert r.json()["category"] == "Cloud/Software"

    def test_tfidf_nb_travel(self):
        r = client.post("/predict?model=tfidf_nb",
                        json={"text": "Air India flight tickets Mumbai to Delhi"})
        assert r.status_code == 200
        assert r.json()["category"] == "Travel"

    def test_tfidf_nb_utilities(self):
        r = client.post("/predict?model=tfidf_nb",
                        json={"text": "Electricity bill for office premises March"})
        assert r.status_code == 200
        assert r.json()["category"] == "Utilities"

    def test_tfidf_nb_confidence_range(self):
        r = client.post("/predict?model=tfidf_nb", json={"text": "HP printer toner cartridge"})
        assert 0.0 <= r.json()["confidence"] <= 1.0

    def test_invalid_model_returns_400(self):
        r = client.post("/predict?model=random_forest", json={"text": "AWS billing"})
        assert r.status_code == 400

    def test_invalid_model_error_message(self):
        r = client.post("/predict?model=xyz", json={"text": "any text"})
        assert "Invalid model" in r.json()["detail"]

    @pytest.mark.slow
    def test_transformer_model(self):
        """Requires HuggingFace model download (~80MB). Marked slow."""
        r = client.post("/predict?model=transformer",
                        json={"text": "AWS monthly cloud hosting bill"})
        assert r.status_code == 200
        assert r.json()["category"] == "Cloud/Software"


# ── /predict – input validation ───────────────────────────────────────────────

class TestPredictValidation:
    def test_empty_text_returns_422(self):
        assert client.post("/predict", json={"text": ""}).status_code == 422

    def test_whitespace_only_text_returns_422(self):
        assert client.post("/predict", json={"text": "   "}).status_code == 422

    def test_missing_text_field_returns_422(self):
        assert client.post("/predict", json={}).status_code == 422

    def test_extra_fields_are_ignored(self):
        r = client.post("/predict", json={"text": "AWS billing", "extra": "ignored"})
        assert r.status_code == 200

    def test_long_text_is_handled(self):
        r = client.post("/predict", json={"text": "AWS cloud hosting " * 100})
        assert r.status_code == 200
        assert r.json()["category"] == "Cloud/Software"

