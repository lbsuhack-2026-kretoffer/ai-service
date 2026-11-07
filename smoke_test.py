"""Симуляционный тест FastAPI-сервиса (с данными бэкенда)."""
import sys
from fastapi.testclient import TestClient

sys.path.insert(0, ".")

import main  # noqa: E402

main.load_models()

print("Models loaded:")
print(f"  standard_global={main.standard_global_model is not None}")
print(f"  buffet={main.buffet_model is not None}")
print(f"  schools={list(main.school_models.keys())}")

client = TestClient(main.app)

# 1. Standard prediction for school-a (fine-tuned модель)
r = client.post("/predict/standard", json={
    "school_id": "school-a", "age_group": "middle", "month": 10,
    "day_of_month": 15, "day_of_week": 2, "is_pre_holiday": 0,
    "is_pre_vacation": 0, "total_students": 300, "opt_outs": 24,
    "present_today": 276, "present_yesterday": 270,
})
print(f"\n[1] Standard /school-a: status={r.status_code}, value={r.json()}")
assert r.status_code == 200

# 2. Standard prediction for school-b (fine-tuned модель)
r = client.post("/predict/standard", json={
    "school_id": "school-b", "age_group": "junior", "month": 10,
    "day_of_month": 15, "day_of_week": 2, "is_pre_holiday": 0,
    "is_pre_vacation": 0, "total_students": 200, "opt_outs": 16,
    "present_today": 184, "present_yesterday": 180,
})
print(f"[2] Standard /school-b: status={r.status_code}, value={r.json()}")
assert r.status_code == 200

# 3. Standard prediction for unknown school (fallback на global)
r = client.post("/predict/standard", json={
    "school_id": "school-999", "age_group": "senior", "month": 3,
    "day_of_month": 1, "day_of_week": 1, "is_pre_holiday": 0,
    "is_pre_vacation": 0, "total_students": 800, "opt_outs": 30,
    "present_today": 770, "present_yesterday": 760,
})
print(f"[3] Standard /school-999 (fallback): status={r.status_code}, value={r.json()}")
assert r.status_code == 200

# 4. ML Service /predict (интеграция с бэкендом, fixed meal plan)
r = client.post("/predict", json={
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
})
print(f"[4] ML /predict (school-a, fixed): status={r.status_code}, body={r.json()}")
assert r.status_code == 200
body = r.json()
assert "total_portions" in body

# 5. ML Service /predict (canteen meal plan)
r = client.post("/predict", json={
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
})
print(f"[5] ML /predict (school-b, canteen): status={r.status_code}, body={r.json()}")
assert r.status_code == 200
body = r.json()
assert "total_portions" in body

# 6. Buffet prediction (прямой API)
r = client.post("/predict/buffet", json={
    "school_id": "school-b", "age_group": "middle", "month": 10,
    "day_of_month": 15, "day_of_week": 2, "is_pre_holiday": 0,
    "is_pre_vacation": 0, "total_students": 200, "opt_outs": 16,
    "present_today": 184, "present_yesterday": 180,
    "sold_hot_yesterday": 70, "sold_salad_yesterday": 30,
    "sold_bakery_yesterday": 45,
})
print(f"[6] Buffet /school-b: status={r.status_code}, body={r.json()}")
assert r.status_code == 200
body = r.json()
assert "hot" in body and "salad" in body and "bakery" in body

# 7. Validation error
r = client.post("/predict/standard", json={
    "school_id": "school-a", "age_group": "invalid", "month": 10,
    "day_of_month": 15, "day_of_week": 2, "is_pre_holiday": 0,
    "is_pre_vacation": 0, "total_students": 300, "opt_outs": 24,
    "present_today": 276, "present_yesterday": 270,
})
print(f"[7] Validation (bad age_group): status={r.status_code}")
assert r.status_code == 422

print("\n=== ALL TESTS PASSED ===")