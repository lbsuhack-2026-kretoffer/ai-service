"""Тесты ai-service: предсказания /predict/standard, /predict/buffet, /predict."""

from __future__ import annotations

from fastapi.testclient import TestClient

STANDARD_PAYLOAD = {
    "school_id": "school-a",
    "age_group": "middle",
    "month": 10,
    "day_of_month": 15,
    "day_of_week": 2,
    "is_pre_holiday": 0,
    "is_pre_vacation": 0,
    "is_holiday_calendar": 0,
    "main_dish": None,
    "total_students": 300,
    "opt_outs": 24,
    "present_today": 276,
    "present_yesterday": 270,
    "present_t_minus_2": 268,
    "present_t_minus_3": 265,
    "present_avg_3d": 272.0,
    "present_avg_7d": 268.0,
    "present_trend": 6.0,
    "present_same_dow_last_week": 274,
    "temp_celsius": 12.0,
    "precipitation_mm": 0.0,
    "is_rainy": 0,
    "is_cold": 0,
}

MULTI_DAY_PAYLOAD = {
    "school_id": "school-a",
    "age_group": "middle",
    "total_students": 300,
    "opt_outs": 24,
    "present_today": 276,
    "present_yesterday": 270,
    "present_t_minus_2": 268,
    "present_t_minus_3": 265,
    "present_avg_3d": 272.0,
    "present_avg_7d": 268.0,
    "present_trend": 6.0,
    "present_same_dow_last_week": 274,
    "days": [
        {
            "date": "2026-10-16",
            "month": 10,
            "day_of_month": 16,
            "day_of_week": 3,
            "is_pre_holiday": 0,
            "is_pre_vacation": 0,
            "is_holiday_calendar": 0,
            "main_dish": None,
            "total_students": 300,
            "opt_outs": 24,
            "present_today": 276,
            "present_yesterday": 270,
            "present_t_minus_2": 268,
            "present_t_minus_3": 265,
            "present_avg_3d": 272.0,
            "present_avg_7d": 268.0,
            "present_trend": 6.0,
            "present_same_dow_last_week": 274,
            "temp_celsius": 11.0,
            "precipitation_mm": 0.0,
            "is_rainy": 0,
            "is_cold": 0,
        },
        {
            "date": "2026-10-17",
            "month": 10,
            "day_of_month": 17,
            "day_of_week": 4,
            "is_pre_holiday": 1,
            "is_pre_vacation": 0,
            "is_holiday_calendar": 0,
            "main_dish": None,
            "total_students": 300,
            "opt_outs": 24,
            "present_today": 276,
            "present_yesterday": 270,
            "present_t_minus_2": 268,
            "present_t_minus_3": 265,
            "present_avg_3d": 272.0,
            "present_avg_7d": 268.0,
            "present_trend": 6.0,
            "present_same_dow_last_week": 274,
            "temp_celsius": 10.0,
            "precipitation_mm": 2.0,
            "is_rainy": 1,
            "is_cold": 0,
        },
        {
            "date": "2026-10-19",
            "month": 10,
            "day_of_month": 19,
            "day_of_week": 0,
            "is_pre_holiday": 0,
            "is_pre_vacation": 0,
            "is_holiday_calendar": 0,
            "main_dish": None,
            "total_students": 300,
            "opt_outs": 24,
            "present_today": 276,
            "present_yesterday": 270,
            "present_t_minus_2": 268,
            "present_t_minus_3": 265,
            "present_avg_3d": 272.0,
            "present_avg_7d": 268.0,
            "present_trend": 6.0,
            "present_same_dow_last_week": 274,
            "temp_celsius": 13.0,
            "precipitation_mm": 0.0,
            "is_rainy": 0,
            "is_cold": 0,
        },
    ],
}

BUFFET_PAYLOAD = {
    "school_id": "school-b",
    "age_group": "middle",
    "month": 10,
    "day_of_month": 15,
    "day_of_week": 2,
    "is_pre_holiday": 0,
    "is_pre_vacation": 0,
    "is_holiday_calendar": 0,
    "main_dish": None,
    "total_students": 200,
    "opt_outs": 16,
    "present_today": 184,
    "present_yesterday": 180,
    "present_t_minus_2": 176,
    "present_t_minus_3": 172,
    "present_avg_3d": 182.0,
    "present_avg_7d": 178.0,
    "present_trend": 4.0,
    "present_same_dow_last_week": 182,
    "temp_celsius": 12.0,
    "precipitation_mm": 0.0,
    "is_rainy": 0,
    "is_cold": 0,
    "sold_hot_yesterday": 70,
    "sold_salad_yesterday": 30,
    "sold_bakery_yesterday": 45,
}


def test_predict_standard_valid(client: TestClient):
    """Валидный вход /predict/standard — 200 и число (float)."""
    response = client.post("/predict/standard", json=STANDARD_PAYLOAD)
    assert response.status_code == 200, response.text
    body = response.json()
    assert "predicted_portions" in body
    assert "confidence" in body
    assert isinstance(body["predicted_portions"], float)
    assert isinstance(body["confidence"], float)


def test_predict_standard_unknown_school_falls_back_to_global(client: TestClient):
    """Неизвестная школа использует глобальную модель — 200."""
    payload = dict(STANDARD_PAYLOAD, school_id="school-999")
    response = client.post("/predict/standard", json=payload)
    assert response.status_code == 200, response.text
    body = response.json()
    assert isinstance(body["predicted_portions"], float)


def test_predict_standard_invalid_age_group_422(client: TestClient):
    """Некорректный age_group — 422."""
    payload = dict(STANDARD_PAYLOAD, age_group="invalid")
    response = client.post("/predict/standard", json=payload)
    assert response.status_code == 422


def test_predict_standard_missing_field_422(client: TestClient):
    """Отсутствующее обязательное поле — 422."""
    payload = dict(STANDARD_PAYLOAD)
    del payload["present_today"]
    response = client.post("/predict/standard", json=payload)
    assert response.status_code == 422


def test_predict_multi_valid(client: TestClient):
    """Валидный вход /predict/multi — 200 со списком предсказаний."""
    response = client.post("/predict/multi", json=MULTI_DAY_PAYLOAD)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["school_id"] == "school-a"
    assert "predictions" in body
    assert len(body["predictions"]) == len(MULTI_DAY_PAYLOAD["days"])
    for p in body["predictions"]:
        assert "date" in p
        assert "predicted_portions" in p
        assert "confidence" in p


def test_predict_multi_too_many_days_422(client: TestClient):
    """Более 5 дней в /predict/multi — 422."""
    payload = dict(MULTI_DAY_PAYLOAD)
    extra_days = list(payload["days"])
    for i in range(3):
        extra_days.append(dict(extra_days[-1], date=f"2026-10-{20+i}"))
    payload["days"] = extra_days
    response = client.post("/predict/multi", json=payload)
    assert response.status_code == 422


def test_predict_buffet_valid(client: TestClient):
    """Валидный вход /predict/buffet — 200 с categories {hot, salad, bakery}."""
    response = client.post("/predict/buffet", json=BUFFET_PAYLOAD)
    assert response.status_code == 200, response.text
    body = response.json()
    assert "categories" in body
    assert set(body["categories"].keys()) == {"hot", "salad", "bakery"}
    assert all(isinstance(v, float) for v in body["categories"].values())


def test_predict_buffet_invalid_age_group_422(client: TestClient):
    """Некорректный age_group для буфета — 422."""
    payload = dict(BUFFET_PAYLOAD, age_group="invalid")
    response = client.post("/predict/buffet", json=payload)
    assert response.status_code == 422


def test_predict_buffet_negative_sales_422(client: TestClient):
    """Отрицательные продажи для буфета — 422."""
    payload = dict(BUFFET_PAYLOAD, sold_hot_yesterday=-5)
    response = client.post("/predict/buffet", json=payload)
    assert response.status_code == 422


ML_PAYLOAD_FIXED = {
    "school_id": "school-a",
    "school_name": "Гимназия №12",
    "date": "2026-10-16",
    "meal_plan": "fixed",
    "buffer_percent": 10,
    "total_students": 300,
    "opt_outs": 24,
    "absent": 0,
    "present": 276,
    "total_portions": 304,
    "by_age_group": [],
    "by_category": [],
    "marked_classes": 3,
    "total_classes": 3,
    "based_on_date": "2026-10-15",
    "confidence": 0.8,
    "forecast_days": 5,
    "main_dish": None,
    "weather": {
        "2026-10-16": {"temp_celsius": 11.0, "precipitation_mm": 0.0, "is_rainy": 0, "is_cold": 0},
        "2026-10-17": {"temp_celsius": 10.0, "precipitation_mm": 2.0, "is_rainy": 1, "is_cold": 0},
    },
    "present_avg_3d": 272.0,
    "present_avg_7d": 268.0,
    "present_trend": 6.0,
    "present_yesterday": 270,
    "present_t_minus_2": 268,
    "present_t_minus_3": 265,
    "present_same_dow_last_week": 274,
    "is_holiday_calendar": 0,
}

ML_PAYLOAD_CANTEEN = {
    "school_id": "school-b",
    "school_name": "Средняя школа №34",
    "date": "2026-10-16",
    "meal_plan": "canteen",
    "buffer_percent": 8,
    "total_students": 200,
    "opt_outs": 16,
    "absent": 0,
    "present": 184,
    "total_portions": 199,
    "by_age_group": [],
    "by_category": [],
    "marked_classes": 2,
    "total_classes": 2,
    "yesterday_total_qty": 88,
    "based_on_date": "2026-10-15",
    "confidence": 0.8,
    "forecast_days": 5,
    "main_dish": None,
    "weather": {
        "2026-10-16": {"temp_celsius": 11.0, "precipitation_mm": 0.0, "is_rainy": 0, "is_cold": 0},
    },
    "present_avg_3d": 182.0,
    "present_avg_7d": 178.0,
    "present_trend": 4.0,
    "present_yesterday": 180,
    "present_t_minus_2": 176,
    "present_t_minus_3": 172,
    "present_same_dow_last_week": 182,
    "is_holiday_calendar": 0,
}


def test_ml_predict_fixed(client: TestClient):
    """POST /predict (ML Service API), meal_plan=fixed — {total_portions, confidence, forecast_days}."""
    response = client.post("/predict", json=ML_PAYLOAD_FIXED)
    assert response.status_code == 200, response.text
    body = response.json()
    assert "total_portions" in body
    assert "confidence" in body
    assert isinstance(body["total_portions"], int)
    assert isinstance(body["confidence"], float)
    assert "forecast_days" in body
    assert isinstance(body["forecast_days"], list)
    assert "model_used" in body


def test_ml_predict_canteen(client: TestClient):
    """POST /predict (ML Service API), meal_plan=canteen — {total_portions, confidence}."""
    response = client.post("/predict", json=ML_PAYLOAD_CANTEEN)
    assert response.status_code == 200, response.text
    body = response.json()
    assert "total_portions" in body
    assert "confidence" in body
    assert isinstance(body["total_portions"], int)
    assert isinstance(body["confidence"], float)
    assert "model_used" in body


def test_ml_predict_missing_required_field_422(client: TestClient):
    """Отсутствующее обязательное поле в /predict — 422."""
    payload = dict(ML_PAYLOAD_FIXED)
    del payload["present"]
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
