"""Симуляционный тест FastAPI-сервиса (с данными бэкенда)."""
import sys
from fastapi.testclient import TestClient

sys.path.insert(0, ".")

import main  # noqa: E402

main.load_models()

print("Models loaded:")
print(f"  standard_global={bool(main.standard_global_models)}")
print(f"  buffet={bool(main.buffet_models)}")
print(f"  schools={list(main.school_models.keys())}")

BASE = {
    "is_holiday_calendar": 0,
    "main_dish": None,
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

client = TestClient(main.app)

# 1. Standard prediction for school-a (fine-tuned модель)
r = client.post("/predict/standard", json={
    "school_id": "school-a", "age_group": "middle", "month": 10,
    "day_of_month": 15, "day_of_week": 2, "is_pre_holiday": 0,
    "is_pre_vacation": 0,
    "total_students": 300, "opt_outs": 24,
    "present_today": 276, "present_yesterday": 270,
    **BASE,
})
print(f"\n[1] Standard /school-a: status={r.status_code}, value={r.json()}")
assert r.status_code == 200

# 2. Standard prediction for school-b (fine-tuned модель)
r = client.post("/predict/standard", json={
    "school_id": "school-b", "age_group": "junior", "month": 10,
    "day_of_month": 15, "day_of_week": 2, "is_pre_holiday": 0,
    "is_pre_vacation": 0,
    "total_students": 200, "opt_outs": 16,
    "present_today": 184, "present_yesterday": 180,
    **{**BASE, "present_t_minus_2": 176, "present_t_minus_3": 172,
       "present_avg_3d": 182.0, "present_avg_7d": 178.0,
       "present_trend": 4.0, "present_same_dow_last_week": 182},
})
print(f"[2] Standard /school-b: status={r.status_code}, value={r.json()}")
assert r.status_code == 200

# 3. Standard prediction for unknown school (fallback на global)
r = client.post("/predict/standard", json={
    "school_id": "school-999", "age_group": "senior", "month": 3,
    "day_of_month": 1, "day_of_week": 1, "is_pre_holiday": 0,
    "is_pre_vacation": 0,
    "total_students": 800, "opt_outs": 30,
    "present_today": 770, "present_yesterday": 760,
    **{**BASE, "present_t_minus_2": 760, "present_t_minus_3": 755,
       "present_avg_3d": 765.0, "present_avg_7d": 762.0,
       "present_trend": 10.0, "present_same_dow_last_week": 768},
})
print(f"[3] Standard /school-999 (fallback): status={r.status_code}, value={r.json()}")
assert r.status_code == 200

# 4. ML Service /predict (интеграция с бэкендом, fixed meal plan)
r = client.post("/predict", json={
    "school_id": "school-a",
    "school_name": "Гимназия №12",
    "age_group": "middle",
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
})
print(f"[4] ML /predict (school-a, fixed): status={r.status_code}, body={r.json()}")
assert r.status_code == 200
body = r.json()
assert "total_portions" in body

# 5. ML Service /predict (canteen meal plan)
r = client.post("/predict", json={
    "school_id": "school-b",
    "school_name": "Средняя школа №34",
    "age_group": "middle",
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
})
print(f"[5] ML /predict (school-b, canteen): status={r.status_code}, body={r.json()}")
assert r.status_code == 200
body = r.json()
assert "total_portions" in body
assert "categories" in body

# 6. Buffet prediction (прямой API)
r = client.post("/predict/buffet", json={
    "school_id": "school-b", "age_group": "middle", "month": 10,
    "day_of_month": 15, "day_of_week": 2, "is_pre_holiday": 0,
    "is_pre_vacation": 0,
    "total_students": 200, "opt_outs": 16,
    "present_today": 184, "present_yesterday": 180,
    "sold_hot_yesterday": 70, "sold_salad_yesterday": 30,
    "sold_bakery_yesterday": 45,
    **{**BASE, "present_t_minus_2": 176, "present_t_minus_3": 172,
       "present_avg_3d": 182.0, "present_avg_7d": 178.0,
       "present_trend": 4.0, "present_same_dow_last_week": 182},
})
print(f"[6] Buffet /school-b: status={r.status_code}, body={r.json()}")
assert r.status_code == 200
body = r.json()
assert "categories" in body
assert set(body["categories"].keys()) == {"hot", "salad", "bakery"}

# 7. Validation error
r = client.post("/predict/standard", json={
    "school_id": "school-a", "age_group": "invalid", "month": 10,
    "day_of_month": 15, "day_of_week": 2, "is_pre_holiday": 0,
    "is_pre_vacation": 0,
    "total_students": 300, "opt_outs": 24,
    "present_today": 276, "present_yesterday": 270,
    **BASE,
})
print(f"[7] Validation (bad age_group): status={r.status_code}")
assert r.status_code == 422

print("\n=== ALL SMOKE TESTS PASSED ===")
