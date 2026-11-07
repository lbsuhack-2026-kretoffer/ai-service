"""Тесты ai-service: загрузка моделей при старте и обработка отсутствующей модели (v2.0)."""

from __future__ import annotations

from pathlib import Path

import pytest

import main


def test_models_loaded_on_startup(client):
    """Модели загружены: глобальные по горизонтам, буфетная, escuela-a, escuela-b."""
    assert main.standard_global_models, "глобальные модели не загружены"
    assert all(h in main.standard_global_models for h in [1, 2, 3, 4, 5]), "не все горизонты загружены"
    assert main.buffet_model is not None, "буфетная модель не загружена"
    assert "school-a" in main.school_models, "модель school-a не загружена"
    assert "school-b" in main.school_models, "модель school-b не загружена"


def test_model_files_exist():
    """Файлы .cbm существуют по ожидаемым путям."""
    assert main.STANDARD_GLOBAL_MODEL_PATHS[1].exists()
    assert main.BUFFET_MODEL_PATH.exists()
    assert any(p.name == "school-a_model_h1.cbm" for p in main.MODELS_DIR.glob("*_model_h*.cbm"))
    assert any(p.name == "school-b_model_h1.cbm" for p in main.MODELS_DIR.glob("*_model_h*.cbm"))


def test_missing_global_model_raises_clear_error(monkeypatch):
    """Отсутствие глобальной модели — понятная ошибка (FileNotFoundError)."""
    monkeypatch.setattr(
        main,
        "STANDARD_GLOBAL_MODEL_PATHS",
        {h: Path(f"/nonexistent/standard_global_model_h{h}.cbm") for h in [1, 2, 3, 4, 5]},
    )
    monkeypatch.setattr(main, "STANDARD_GLOBAL_MODEL_PATH_FALLBACK", Path("/nonexistent/standard_global_model_h1.cbm"))
    with pytest.raises(FileNotFoundError, match="Глобальные модели не найдены"):
        main.load_models()


def test_prediction_without_model_raises_runtime_error(monkeypatch):
    """Предсказание без загруженной модели — ошибка RuntimeError, а не пустой ответ."""
    payload = {
        "school_id": "school-x", "age_group": "middle", "month": 10,
        "day_of_month": 15, "day_of_week": 2, "is_pre_holiday": 0,
        "is_pre_vacation": 0, "main_dish": None,
        "total_students": 100, "opt_outs": 5,
        "present_today": 95, "present_yesterday": 90,
        "present_avg_3d": 92.0, "present_avg_7d": 90.0, "present_trend": 5.0,
        "temp_celsius": 12.0, "precipitation_mm": 0.0, "is_rainy": 0, "is_cold": 0,
    }
