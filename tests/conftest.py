"""Фикстуры для тестов ai-service (v2.1 — многогоризонт, единый источник фичей).

Модельные файлы .cbm в репозитории отсутствуют, поэтому перед тестами
обучаются минимальные заглушки-модели CatBoost во временной директории и
подменяются пути в модуле main, после чего модели загружаются через
load_models() (аналогично событию старта приложения).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest
from catboost import CatBoostRegressor, Pool
from fastapi.testclient import TestClient

import main
from features import (
    HORIZONS,
    CAT_FEATURES,
    BINARY_FEATURES,
    NUMERIC_FEATURES_STANDARD,
    NUMERIC_FEATURES_BUFFET,
    STANDARD_INPUT_FEATURES,
    BUFFET_INPUT_FEATURES,
)


def _train_and_save(features: list[str], path: Path, n_rows: int = 40) -> None:
    """Обучает минимальную модель на синтетических данных и сохраняет .cbm."""
    data = {
        "school_id": ["school-a"] * (n_rows // 2) + ["school-b"] * (n_rows - n_rows // 2),
        "age_group": [["junior", "middle", "senior"][i % 3] for i in range(n_rows)],
        "month": [10] * n_rows,
        "day_of_month": [15] * n_rows,
        "day_of_week": [i % 7 for i in range(n_rows)],
        "is_pre_holiday": [0] * n_rows,
        "is_pre_vacation": [0] * n_rows,
        "is_holiday_calendar": [0] * n_rows,
        "main_dish": ["гречка с котлетой", "макароны по-флотски"] * (n_rows // 2) + ["гречка с котлетой"] * (n_rows % 2),
        "total_students": [300] * n_rows,
        "opt_outs": [24] * n_rows,
        "present_today": [276] * n_rows,
        "present_yesterday": [270] * n_rows,
        "present_t_minus_2": [268] * n_rows,
        "present_t_minus_3": [265] * n_rows,
        "present_avg_3d": [272.0] * n_rows,
        "present_avg_7d": [268.0] * n_rows,
        "present_trend": [6.0] * n_rows,
        "present_same_dow_last_week": [274.0] * n_rows,
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

    # Стандартные модели для каждого горизонта
    for h in HORIZONS:
        _train_and_save(STANDARD_INPUT_FEATURES, tmp / f"standard_global_model_v2_h{h}.cbm")

    # Буфетные модели для каждого горизонта
    for h in HORIZONS:
        _train_and_save(BUFFET_INPUT_FEATURES, tmp / f"buffet_model_v2_h{h}.cbm")

    # Школьные модели для каждого горизонта
    for school_id in ["school-a", "school-b"]:
        for h in HORIZONS:
            _train_and_save(STANDARD_INPUT_FEATURES, tmp / f"{school_id}_model_v2_h{h}.cbm")

    mp = pytest.MonkeyPatch()
    mp.setattr(main, "MODELS_DIR", tmp)
    mp.setattr(
        main,
        "STANDARD_GLOBAL_MODEL_PATHS",
        {h: tmp / f"standard_global_model_v2_h{h}.cbm" for h in HORIZONS},
    )
    mp.setattr(
        main,
        "BUFFET_MODEL_PATHS",
        {h: tmp / f"buffet_model_v2_h{h}.cbm" for h in HORIZONS},
    )
    main.standard_global_models = {}
    main.buffet_models = {}
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
