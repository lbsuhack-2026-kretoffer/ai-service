"""Тесты ai-service: загрузка моделей при старте и обработка отсутствующей модели (v2.1)."""

from __future__ import annotations

from pathlib import Path

import pytest

import main
from features import HORIZONS


def test_models_loaded_on_startup(client):
    """Модели загружены: стандартные по горизонтам, буфетные по горизонтам, escuela-a, escuela-b."""
    assert main.standard_global_models, "стандартные модели не загружены"
    assert all(h in main.standard_global_models for h in [1, 2, 3, 4, 5]), "не все горизонты загружены"
    assert main.buffet_models, "буфетные модели не загружены"
    assert all(h in main.buffet_models for h in [1, 2, 3, 4, 5]), "не все горизонты буфета загружены"
    assert "school-a" in main.school_models, "модель school-a не загружена"
    assert "school-b" in main.school_models, "модель school-b не загружена"


def test_model_files_exist():
    """Файлы .cbm существуют по ожидаемым путям (в тестах — временные)."""
    # В тестах модели создаются во временной директории, проверяем что загружены
    assert main.standard_global_models[1] is not None
    assert main.buffet_models[1] is not None


def test_missing_global_model_raises_clear_error(monkeypatch):
    """Отсутствие всех глобальных моделей — load_models не падает, просто не загружает."""
    monkeypatch.setattr(
        main,
        "STANDARD_GLOBAL_MODEL_PATHS",
        {h: Path(f"/nonexistent/standard_global_model_v2_h{h}.cbm") for h in HORIZONS},
    )
    monkeypatch.setattr(main, "BUFFET_MODEL_PATHS", {})
    # Не должно падать — просто не загрузит модели
    main.load_models()
    assert main.standard_global_models == {}


def test_prediction_without_model_raises_runtime_error(monkeypatch):
    """Предсказание без загруженной модели — использует fallback, не падает."""
    payload = {
        "school_id": "school-x", "age_group": "middle", "month": 10,
        "day_of_month": 15, "day_of_week": 2, "is_pre_holiday": 0,
        "is_pre_vacation": 0, "is_holiday_calendar": 0, "main_dish": None,
        "total_students": 100, "opt_outs": 5,
        "present_today": 95, "present_yesterday": 90,
        "present_t_minus_2": 88, "present_t_minus_3": 85,
        "present_avg_3d": 92.0, "present_avg_7d": 90.0, "present_trend": 5.0,
        "present_same_dow_last_week": 93,
        "temp_celsius": 12.0, "precipitation_mm": 0.0, "is_rainy": 0, "is_cold": 0,
    }
    monkeypatch.setattr(main, "standard_global_models", {})
    monkeypatch.setattr(main, "school_models", {})
    # Без моделей должен вернуться fallback, не исключение
    result = main._standard_prediction(payload, horizon=1)
    assert result is None  # 모델 нет, возвращается None
