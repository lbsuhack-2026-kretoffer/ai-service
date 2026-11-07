"""Фикстуры для тестов ai-service (v2.0 — многогоризонт).

Модельные файлы .cbm в репозитории отсутствуют, поэтому перед тестами
обучаются минимальные заглушки-модели CatBoost во временной директории и
подменяются пути в модуле main, после которого модели загружаются через
load_models() (аналогично событию старта приложения).
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest
from catboost import CatBoostRegressor, Pool
from fastapi.testclient import TestClient

import main

# Списки фичей из main.py (v2.0)
CAT_FEATURES = [
    "school_id",
    "age_group",
    "month",
    "day_of_month",
    "day_of_week",
    "main_dish",
]
BINARY_FEATURES = [
    "is_pre_holiday",
    "is_pre_vacation",
    "is_rainy",
    "is_cold",
]
NUMERIC_STANDARD = [
    "total_students",
    "opt_outs",
    "present_today",
    "present_yesterday",
    "present_avg_3d",
    "present_avg_7d",
    "present_trend",
    "temp_celsius",
    "precipitation_mm",
]
STANDARD_FEATURES = CAT_FEATURES + BINARY_FEATURES + NUMERIC_STANDARD
BUFFET_FEATURES = STANDARD_FEATURES + [
    "sold_hot_yesterday",
    "sold_salad_yesterday",
    "sold_bakery_yesterday",
]

HORIZONS = [1, 2, 3, 4, 5]


def _train_and_save(features: list[str], path: Path, n_rows: int = 30) -> None:
    """Обучает минимальную модель на синтетических данных и сохраняет .cbm."""
    data = {
        "school_id": ["school-a"] * (n_rows // 2) + ["school-b"] * (n_rows - n_rows // 2),
        "age_group": [["junior", "middle", "senior"][i % 3] for i in range(n_rows)],
        "month": [10] * n_rows,
        "day_of_month": [15] * n_rows,
        "day_of_week": [i % 7 for i in range(n_rows)],
        "is_pre_holiday": [0] * n_rows,
        "is_pre_vacation": [0] * n_rows,
        "main_dish": ["гречка с котлетой", "макароны по-флотски"] * (n_rows // 2) + ["гречка с котлетой"] * (n_rows % 2),
        "total_students": [300] * n_rows,
        "opt_outs": [24] * n_rows,
        "present_today": [276] * n_rows,
        "present_yesterday": [270] * n_rows,
        "present_avg_3d": [272.0] * n_rows,
        "present_avg_7d": [268.0] * n_rows,
        "present_trend": [6.0] * n_rows,
        "temp_celsius": [12.0] * n_rows,
        "precipitation_mm": [0.0] * n_rows,
        "is_rainy": [0] * n_rows,
        "is_cold": [0] * n_rows,
    }
    if "sold_hot_yesterday" in features:
        data["sold_hot_yesterday"] = [70] * n_rows
        data["sold_salad_yesterday"] = [30] * n_rows
        data["sold_bakery_yesterday"] = [45] * n_rows
    df = pd.DataFrame(data)
    if "sold_hot_yesterday" in features:
        y = pd.DataFrame({
            "hot": [70.0 + i for i in range(n_rows)],
            "salad": [30.0 + i for i in range(n_rows)],
            "bakery": [45.0 + i for i in range(n_rows)],
        })
        model = CatBoostRegressor(iterations=5, loss_function="MultiRMSE", verbose=False)
    else:
        y = pd.Series(range(n_rows), dtype=float)
        model = CatBoostRegressor(iterations=5, verbose=False)
    X = df[features]
    for c in CAT_FEATURES:
        if c in X.columns:
            X[c] = X[c].astype(str)
    cat_positions = [list(X.columns).index(c) for c in CAT_FEATURES if c in X.columns]
    model.fit(Pool(X, y, cat_features=cat_positions), verbose=False)
    model.save_model(str(path))


@pytest.fixture(scope="session", autouse=True)
def _dummy_models():
    """Создаёт временные .cbm-модели для всех горизонтов, подменяет пути в main и загружает модели."""
    tmp = Path(tempfile.mkdtemp(prefix="aiservice_test_models_"))

    # Глобальные модели для каждого горизонта
    for h in HORIZONS:
        _train_and_save(STANDARD_FEATURES, tmp / f"standard_global_model_h{h}.cbm")

    # Буфетная модель
    _train_and_save(BUFFET_FEATURES, tmp / "buffet_model.cbm")

    # Школьные модели для каждого горизонта
    for school_id in ["school-a", "school-b"]:
        for h in HORIZONS:
            _train_and_save(STANDARD_FEATURES, tmp / f"{school_id}_model_h{h}.cbm")

    mp = pytest.MonkeyPatch()
    mp.setattr(main, "MODELS_DIR", tmp)
    # STANDARD_GLOBAL_MODEL_PATHS — словарь horizon -> Path
    mp.setattr(
        main,
        "STANDARD_GLOBAL_MODEL_PATHS",
        {h: tmp / f"standard_global_model_h{h}.cbm" for h in HORIZONS},
    )
    mp.setattr(main, "STANDARD_GLOBAL_MODEL_PATH_FALLBACK", tmp / "standard_global_model_h1.cbm")
    mp.setattr(main, "BUFFET_MODEL_PATH", tmp / "buffet_model.cbm")
    main.standard_global_models = {}
    main.buffet_model = None
    main.school_models = {}
    main.load_models()
    yield
    mp.undo()
    import shutil
    shutil.rmtree(tmp, ignore_errors=True)


@pytest.fixture
def client():
    """Тестовый клиент FastAPI (модели уже загружены фикстурой выше)."""
    return TestClient(main.app)
