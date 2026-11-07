#!/usr/bin/env python3
"""
FastAPI backend сервис для предсказания школьного питания (версия 2.1).

Сервис:
- Загружает модели при старте:
  * Глобальные модели по горизонтам: standard_global_model_v2_h{h}.cbm (h=1..5)
  * Дообученные модели школ: school_{id}_model_v2_h{h}.cbm (Transfer Learning)
  * Буфетные модели по горизонтам: buffet_model_v2_h{h}.cbm (h=1..5)
- Эндпоинты:
  1. POST /predict/standard — стандартное питание на 1 день вперёд
  2. POST /predict/multi — стандартное питание на 5 дней вперёд
  3. POST /predict/buffet — шведский стол (мультитаргет, многогоризонт)
  4. POST /predict — интеграция с бэкендом (ML Service API)

Все списки фичей импортируются из features.py — единый источник правды.
При отсутствии моделей сервис не падает, а использует базовую формулу
present * (1 + buffer_percent / 100) как fallback.
"""

from __future__ import annotations

import logging
import re
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from catboost import CatBoostRegressor, Pool
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator

# --- Единый источник фичей ---
from features import (
    HORIZONS,
    CAT_FEATURES,
    BINARY_FEATURES,
    NUMERIC_FEATURES_STANDARD,
    NUMERIC_FEATURES_BUFFET,
    STANDARD_INPUT_FEATURES,
    BUFFET_INPUT_FEATURES,
    TARGET_STANDARD,
    TARGETS_BUFFET,
    AGE_GROUPS,
    COLD_THRESHOLD,
    BUFFET_CATEGORIES,
    buffet_ratio_for,
    get_holiday_calendar,
)

# ---------------------------------------------------------------------------
# Пути к моделям (версия v2)
# ---------------------------------------------------------------------------
MODELS_DIR = Path(__file__).parent / "models"

MODEL_VERSION = "v2"

STANDARD_GLOBAL_MODEL_PATHS = {
    h: MODELS_DIR / f"standard_global_model_{MODEL_VERSION}_h{h}.cbm" for h in HORIZONS
}
BUFFET_MODEL_PATHS = {
    h: MODELS_DIR / f"buffet_model_{MODEL_VERSION}_h{h}.cbm" for h in HORIZONS
}

# ---------------------------------------------------------------------------
# Глобальные переменные моделей
# ---------------------------------------------------------------------------
standard_global_models: Dict[int, CatBoostRegressor] = {}  # horizon -> model
buffet_models: Dict[int, CatBoostRegressor] = {}  # horizon -> model (global)
school_models: Dict[str, Dict[int, CatBoostRegressor]] = {}  # school_id -> {horizon: model} (standard)
school_buffet_models: Dict[str, Dict[int, CatBoostRegressor]] = {}  # school_id -> {horizon: model} (buffet)

logger = logging.getLogger("uvicorn.error")


# ---------------------------------------------------------------------------
# Загрузка моделей
# ---------------------------------------------------------------------------
def _load_single_model(path: Path) -> CatBoostRegressor | None:
    """Загружает одну модель по пути, возвращает None если файл не найден."""
    if not path.exists():
        return None
    model = CatBoostRegressor()
    model.load_model(str(path))
    return model


def load_models() -> None:
    """Загрузка всех моделей при старте приложения.

    Не падает при отсутствии моделей — просто не загружает их.
    Инференс использует базовую формулу как fallback.
    """
    global standard_global_models, buffet_models, school_models

    new_global_models: Dict[int, CatBoostRegressor] = {}
    new_buffet_models: Dict[int, CatBoostRegressor] = {}
    new_school_models: Dict[str, Dict[int, CatBoostRegressor]] = {}
    new_school_buffet_models: Dict[str, Dict[int, CatBoostRegressor]] = {}

    # Глобальные модели по горизонтам
    for h in HORIZONS:
        path = STANDARD_GLOBAL_MODEL_PATHS.get(h)
        model = _load_single_model(path) if path else None
        if model is not None:
            new_global_models[h] = model

    # Буфетные модели по горизонтам
    for h in HORIZONS:
        path = BUFFET_MODEL_PATHS.get(h)
        model = _load_single_model(path) if path else None
        if model is not None:
            new_buffet_models[h] = model

    # Дообученные стандартные модели школ по горизонтам (исключаем буфетные и глобальные)
    pattern = re.compile(r"^(.+)_model_v2_h(\d+)\.cbm$")
    excluded_prefixes = ("standard_global_model", "buffet_model", "standard_model")
    for model_file in MODELS_DIR.glob("*_model_v2_h*.cbm"):
        name = model_file.name
        if any(name.startswith(p) for p in excluded_prefixes):
            continue
        if "_buffet_model_" in name:
            continue  # буфетные школьные модели загружаются отдельно
        match = pattern.match(name)
        if not match:
            continue
        school_id = match.group(1)
        horizon = int(match.group(2))
        model = CatBoostRegressor()
        model.load_model(str(model_file))
        new_school_models.setdefault(school_id, {})[horizon] = model

    # Дообученные буфетные модели школ по горизонтам
    buffet_pattern = re.compile(r"^(.+)_buffet_model_v2_h(\d+)\.cbm$")
    for model_file in MODELS_DIR.glob("*_buffet_model_v2_h*.cbm"):
        name = model_file.name
        match = buffet_pattern.match(name)
        if not match:
            continue
        school_id = match.group(1)
        horizon = int(match.group(2))
        model = CatBoostRegressor()
        model.load_model(str(model_file))
        new_school_buffet_models.setdefault(school_id, {})[horizon] = model

    # Атомарное присваивание globals
    standard_global_models = new_global_models
    buffet_models = new_buffet_models
    school_models = new_school_models
    school_buffet_models = new_school_buffet_models

    logger.info(
        f"Модели загружены: стандартные горизонты {sorted(standard_global_models.keys())}, "
        f"буфетные горизонты {sorted(buffet_models.keys())}, "
        f"школ (standard) {len(school_models)}, "
        f"школ (buffet) {len(school_buffet_models)}"
    )


# ---------------------------------------------------------------------------
# Базовые формулы (fallback когда модели недоступны)
# ---------------------------------------------------------------------------
def _base_formula_portions(present: int, buffer_percent: int) -> int:
    """Базовый расчёт порций без ML-модели."""
    if present <= 0:
        return 0
    return max(0, int(present * (1 + buffer_percent / 100)))


# ---------------------------------------------------------------------------
# Хелперы предсказания
# ---------------------------------------------------------------------------
def _make_prediction_pool(
    input_data: dict,
    feature_names: list[str],
    cat_feature_names: list[str],
) -> Pool:
    """Создает CatBoost Pool из словаря данных для предсказания."""
    df = pd.DataFrame([input_data])

    for col in feature_names:
        if col not in df.columns:
            df[col] = 0

    df = df[feature_names]

    for c in cat_feature_names:
        if c in df.columns:
            df[c] = df[c].fillna("unknown").astype(str)

    cat_positions = [list(df.columns).index(c) for c in cat_feature_names]
    return Pool(df, cat_features=cat_positions)


def _get_school_model(
    school_id: str, horizon: int, model_type: str = "standard"
) -> CatBoostRegressor | None:
    """Возвращает модель для школы и горизонта (с fallback на глобальную)."""
    if model_type == "standard":
        school_horizon_models = school_models.get(school_id, {})
        if horizon in school_horizon_models:
            return school_horizon_models[horizon]
        return standard_global_models.get(horizon)
    else:
        # buffet
        school_horizon_models = school_buffet_models.get(school_id, {})
        if horizon in school_horizon_models:
            return school_horizon_models[horizon]
        return buffet_models.get(horizon)


def _standard_prediction(
    request_data: dict, horizon: int = 1
) -> float | None:
    """Внутренний предсказатель стандартного питания для заданного горизонта.

    Возвращает None если модель не найдена (вызывающий код должен использовать fallback).
    """
    model = _get_school_model(
        request_data.get("school_id", ""), horizon, model_type="standard"
    )
    if model is None:
        return None

    pool = _make_prediction_pool(request_data, STANDARD_INPUT_FEATURES, CAT_FEATURES)
    prediction = model.predict(pool).flatten()[0]

    logger.info(
        f"Standard prediction h={horizon}: school={request_data.get('school_id')}, "
        f"predicted_portions={prediction:.1f}"
    )
    return float(prediction)


def _buffet_prediction(
    request_data: dict, horizon: int = 1
) -> dict | None:
    """Внутренний предсказатель шведского стола для заданного горизонта.

    Возвращает None если модель не найдена.
    """
    model = _get_school_model(
        request_data.get("school_id", ""), horizon, model_type="buffet"
    )
    if model is None:
        return None

    pool = _make_prediction_pool(request_data, BUFFET_INPUT_FEATURES, CAT_FEATURES)
    prediction = model.predict(pool)

    # CatBoost MultiRMSE может вернуть 1D массив при одной строке — приводим к 2D
    if prediction.ndim == 1:
        prediction = prediction.reshape(1, -1)

    n_categories = len(BUFFET_CATEGORIES)
    if prediction.shape[1] != n_categories:
        logger.error(
            f"Буфетная модель h={horizon}: ожидалось {n_categories} предсказаний "
            f"(категории {BUFFET_CATEGORIES}), получено {prediction.shape[1]}"
        )
        return None

    return {
        cat: float(prediction[0][i])
        for i, cat in enumerate(BUFFET_CATEGORIES)
    }


def _calculate_confidence(
    school_id: str,
    horizon: int,
    prediction: float,
    present_today: int,
    total_students: int,
    has_model: bool,
) -> float:
    """
    Расчёт уверенности в предсказании.

    Логика:
    - Базовая уверенность выше если есть модель школы и далёкий горизонт штрафуется
    - Если прогноз сильно отличается от посещаемости сегодня — ниже уверенность
    - Если модель отсутствует (fallback) — уверенность низкая
    - Для далёких горизонтов уверенность légèrement ниже
    """
    if not has_model:
        return 0.50  # fallback — низкая уверенность

    has_school_model = school_id in school_models and horizon in school_models.get(school_id, {})
    base_confidence = 0.88 if has_school_model else 0.80

    horizon_penalty = (horizon - 1) * 0.02

    if present_today > 0:
        deviation = abs(prediction - present_today) / present_today
        deviation_penalty = min(deviation * 0.3, 0.15)
    else:
        deviation_penalty = 0.05

    size_penalty = 0.0 if total_students >= 200 else min((200 - total_students) / 200 * 0.05, 0.05)

    confidence = base_confidence - horizon_penalty - deviation_penalty - size_penalty
    return max(0.55, min(0.95, confidence))


# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Школьный Баланс-Пит - AI сервис предсказания питания",
    description="Сервис для предсказания количества порций питания (1 и 5 дней вперёд)",
    version="2.1.0",
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "version": "2.1.0",
        "model_version": MODEL_VERSION,
        "standard_horizons": sorted(standard_global_models.keys()),
        "buffet_horizons": sorted(buffet_models.keys()),
        "schools": list(school_models.keys()),
    }


@app.on_event("startup")
async def startup_event() -> None:
    logger.info("Запуск приложения...")
    load_models()


# ---------------------------------------------------------------------------
# Pydantic схемы (прямой API)
# ---------------------------------------------------------------------------
class DayFeatures(BaseModel):
    """Признаки для одного целевого дня прогноза."""
    date: str = Field(..., description="Дата в формате ISO (YYYY-MM-DD)")
    month: int = Field(..., ge=1, le=12)
    day_of_month: int = Field(..., ge=1, le=31)
    day_of_week: int = Field(..., ge=0, le=6, description="0=Пн, 6=Вс")
    is_pre_holiday: int = Field(..., ge=0, le=1)
    is_pre_vacation: int = Field(..., ge=0, le=1)
    is_holiday_calendar: int = Field(..., ge=0, le=1, description="Праздник/каникулы из календаря")
    main_dish: Optional[str] = Field(None, description="Основное блюда дня (для fixed-питания)")
    total_students: int = Field(..., ge=1)
    opt_outs: int = Field(..., ge=0)
    present_today: int = Field(..., ge=1)
    present_yesterday: int = Field(..., ge=0)
    present_t_minus_2: int = Field(..., ge=0, description="Посещаемость 2 дня назад")
    present_t_minus_3: int = Field(..., ge=0, description="Посещаемость 3 дня назад")
    present_avg_3d: float = Field(..., ge=0.0, description="Средняя посещаемость за 3 дня")
    present_avg_7d: float = Field(..., ge=0.0, description="Средняя посещаемость за 7 дней")
    present_trend: float = Field(..., description="Тренд посещаемости (сегодня минус вчера)")
    present_same_dow_last_week: int = Field(..., ge=0, description="Посещаемость в этот же день недели на прошлой неделе")
    temp_celsius: float = Field(..., description="Температура воздуха, °C")
    precipitation_mm: float = Field(..., ge=0.0, description="Осадки, мм")
    is_rainy: int = Field(..., ge=0, le=1, description="1 если дождь/снег")
    is_cold: int = Field(..., ge=0, le=1, description="1 если температура ниже порога")

    @validator("is_pre_holiday", "is_pre_vacation", "is_holiday_calendar", "is_rainy", "is_cold")
    def validate_binary(cls, v):
        if v not in [0, 1]:
            raise ValueError("бинарные поля должны быть 0 или 1")
        return v


class StandardPredictionRequest(BaseModel):
    """Схема запроса для предсказания стандартного питания (1 день)."""
    school_id: str = Field(..., description="Идентификатор школы")
    age_group: str = Field(..., description="Возрастная группа: 'junior', 'middle', 'senior'")
    month: int = Field(..., ge=1, le=12)
    day_of_month: int = Field(..., ge=1, le=31)
    day_of_week: int = Field(..., ge=0, le=6)
    is_pre_holiday: int = Field(..., ge=0, le=1)
    is_pre_vacation: int = Field(..., ge=0, le=1)
    is_holiday_calendar: int = Field(..., ge=0, le=1)
    main_dish: Optional[str] = Field(None, description="Основное блюда дня")
    total_students: int = Field(..., ge=1)
    opt_outs: int = Field(..., ge=0)
    present_today: int = Field(..., ge=1)
    present_yesterday: int = Field(..., ge=0)
    present_t_minus_2: int = Field(..., ge=0)
    present_t_minus_3: int = Field(..., ge=0)
    present_avg_3d: float = Field(..., ge=0.0)
    present_avg_7d: float = Field(..., ge=0.0)
    present_trend: float = Field(...)
    present_same_dow_last_week: int = Field(..., ge=0)
    temp_celsius: float = Field(..., description="Температура, °C")
    precipitation_mm: float = Field(..., ge=0.0, description="Осадки, мм")
    is_rainy: int = Field(..., ge=0, le=1)
    is_cold: int = Field(..., ge=0, le=1)

    @validator("age_group")
    def validate_age_group(cls, v):
        if v not in ["junior", "middle", "senior"]:
            raise ValueError("age_group должен быть 'junior', 'middle' или 'senior'")
        return v

    @validator("opt_outs")
    def validate_opt_outs(cls, v, values):
        total = values.get("total_students", 0)
        if v > total:
            raise ValueError("opt_outs не может быть больше total_students")
        return v

    @validator("present_today")
    def validate_present_today(cls, v, values):
        total = values.get("total_students", 0)
        if v > total:
            raise ValueError("present_today не может быть больше total_students")
        return v


class MultiDayPredictionRequest(BaseModel):
    """Схема запроса для предсказания на 5 дней вперёд."""
    school_id: str = Field(..., description="Идентификатор школы")
    age_group: str = Field(..., description="Возрастная группа")
    total_students: int = Field(..., ge=1)
    opt_outs: int = Field(..., ge=0)
    present_today: int = Field(..., ge=1)
    present_yesterday: int = Field(..., ge=0)
    present_t_minus_2: int = Field(..., ge=0)
    present_t_minus_3: int = Field(..., ge=0)
    present_avg_3d: float = Field(..., ge=0.0)
    present_avg_7d: float = Field(..., ge=0.0)
    present_trend: float = Field(...)
    present_same_dow_last_week: int = Field(..., ge=0)
    days: List[DayFeatures] = Field(..., min_length=1, max_length=5, description="Признаки для каждого целевого дня (1-5 дней)")

    @validator("age_group")
    def validate_age_group(cls, v):
        if v not in ["junior", "middle", "senior"]:
            raise ValueError("age_group должен быть 'junior', 'middle' или 'senior'")
        return v

    @validator("opt_outs")
    def validate_opt_outs(cls, v, values):
        total = values.get("total_students", 0)
        if v > total:
            raise ValueError("opt_outs не может быть больше total_students")
        return v

    @validator("present_today")
    def validate_present_today(cls, v, values):
        total = values.get("total_students", 0)
        if v > total:
            raise ValueError("present_today не может быть больше total_students")
        return v


class DayPrediction(BaseModel):
    """Прогноз на один день."""
    date: str
    predicted_portions: float
    confidence: float


class StandardPredictionResponse(BaseModel):
    predicted_portions: float
    confidence: float


class MultiDayPredictionResponse(BaseModel):
    school_id: str
    predictions: List[DayPrediction]


class BuffetPredictionRequest(BaseModel):
    """Схема запроса для предсказания шведского стола."""
    school_id: str
    age_group: str = Field(..., description="junior, middle, senior")
    month: int = Field(..., ge=1, le=12)
    day_of_month: int = Field(..., ge=1, le=31)
    day_of_week: int = Field(..., ge=0, le=6)
    is_pre_holiday: int = Field(..., ge=0, le=1)
    is_pre_vacation: int = Field(..., ge=0, le=1)
    is_holiday_calendar: int = Field(..., ge=0, le=1)
    main_dish: Optional[str] = None
    total_students: int = Field(..., ge=1)
    opt_outs: int = Field(..., ge=0)
    present_today: int = Field(..., ge=1)
    present_yesterday: int = Field(..., ge=0)
    present_t_minus_2: int = Field(..., ge=0)
    present_t_minus_3: int = Field(..., ge=0)
    present_avg_3d: float = Field(..., ge=0.0)
    present_avg_7d: float = Field(..., ge=0.0)
    present_trend: float = Field(...)
    present_same_dow_last_week: int = Field(..., ge=0)
    temp_celsius: float = Field(...)
    precipitation_mm: float = Field(..., ge=0.0)
    is_rainy: int = Field(..., ge=0, le=1)
    is_cold: int = Field(..., ge=0, le=1)
    sold_hot_yesterday: int = Field(..., ge=0)
    sold_salad_yesterday: int = Field(..., ge=0)
    sold_bakery_yesterday: int = Field(..., ge=0)

    @validator("age_group")
    def validate_age_group(cls, v):
        if v not in ["junior", "middle", "senior"]:
            raise ValueError("age_group должен быть 'junior', 'middle' или 'senior'")
        return v

    @validator("is_pre_holiday", "is_pre_vacation", "is_holiday_calendar", "is_rainy", "is_cold")
    def validate_binary(cls, v):
        if v not in [0, 1]:
            raise ValueError("бинарные поля должны быть 0 или 1")
        return v

    @validator("opt_outs")
    def validate_opt_outs(cls, v, values):
        total = values.get("total_students", 0)
        if v > total:
            raise ValueError("opt_outs не может быть больше total_students")
        return v

    @validator("present_today")
    def validate_present_today(cls, v, values):
        total = values.get("total_students", 0)
        if v > total:
            raise ValueError("present_today не может быть больше total_students")
        return v


class BuffetPredictionResponse(BaseModel):
    categories: Dict[str, float]


# ---------------------------------------------------------------------------
# ML Service API (Backend Integration)
# ---------------------------------------------------------------------------
class MLPredictRequest(BaseModel):
    """
    Полoad от бэкенда (ForecastOut из backend/app/schemas.py).
    ml_client.py отправляет этот dict на /predict.
    """
    school_id: str
    school_name: str | None = None
    age_group: str = Field("middle", description="Возрастная группа: junior, middle, senior")
    date: str | date | None = None
    meal_plan: str = "fixed"
    buffer_percent: int = 8
    total_students: int
    opt_outs: int
    absent: int = 0
    present: int
    total_portions: int
    by_age_group: list[dict] = []
    by_category: list[dict] = []
    marked_classes: int = 0
    total_classes: int = 0
    based_on_date: str | date | None = None
    confidence: float = 0.5

    # Новые поля для многогоризонта, погоды и блюда
    forecast_days: int = Field(5, ge=1, le=5, description="Количество дней прогноза вперёд")
    main_dish: str | None = Field(None, description="Основное блюда дня (fixed-питание)")
    weather: dict | None = Field(
        None,
        description="Погода по дням: {ISO_date: {temp_celsius, precipitation_mm, is_rainy, is_cold}}",
    )
    present_avg_3d: float | None = Field(None, description="Средняя посещаемость за 3 дня")
    present_avg_7d: float | None = Field(None, description="Средняя посещаемость за 7 дней")
    present_trend: float | None = Field(None, description="Тренд посещаемости")
    present_yesterday: int | None = Field(None, description="Посещаемость вчера")
    present_t_minus_2: int | None = Field(None, description="Посещаемость 2 дня назад")
    present_t_minus_3: int | None = Field(None, description="Посещаемость 3 дня назад")
    present_same_dow_last_week: int | None = Field(None, description="Посещаемость в этот же день недели на прошлой неделе")
    is_holiday_calendar: int | None = Field(None, description="Праздник/каникулы из календаря")

    @validator("age_group")
    def validate_age_group(cls, v):
        if v not in AGE_GROUPS:
            raise ValueError("age_group должен быть 'junior', 'middle' или 'senior'")
        return v


class MLPredictResponse(BaseModel):
    """Ответ AI-сервиса — скорректированный прогноз."""
    total_portions: int | None = None
    confidence: float | None = None
    forecast_days: list[dict] | None = None  # [{date, predicted_portions, confidence, horizon}, ...]
    model_used: str | None = None  # "ml" или "fallback"
    categories: dict[str, float] | None = None  # прогноз по категориям буфета (для canteen)


# ---------------------------------------------------------------------------
# Эндпоинты (прямой API)
# ---------------------------------------------------------------------------
@app.post("/predict/standard", response_model=StandardPredictionResponse)
async def predict_standard(request: StandardPredictionRequest) -> StandardPredictionResponse:
    """Предсказание количества порций на 1 день вперёд."""
    input_data = {
        "school_id": request.school_id,
        "age_group": request.age_group,
        "month": request.month,
        "day_of_month": request.day_of_month,
        "day_of_week": request.day_of_week,
        "is_pre_holiday": request.is_pre_holiday,
        "is_pre_vacation": request.is_pre_vacation,
        "is_holiday_calendar": request.is_holiday_calendar,
        "main_dish": request.main_dish,
        "total_students": request.total_students,
        "opt_outs": request.opt_outs,
        "present_today": request.present_today,
        "present_yesterday": request.present_yesterday,
        "present_t_minus_2": request.present_t_minus_2,
        "present_t_minus_3": request.present_t_minus_3,
        "present_avg_3d": request.present_avg_3d,
        "present_avg_7d": request.present_avg_7d,
        "present_trend": request.present_trend,
        "present_same_dow_last_week": request.present_same_dow_last_week,
        "temp_celsius": request.temp_celsius,
        "precipitation_mm": request.precipitation_mm,
        "is_rainy": request.is_rainy,
        "is_cold": request.is_cold,
    }
    portions = _standard_prediction(input_data, horizon=1)
    has_model = portions is not None
    if portions is None:
        # Fallback: базовая формула
        portions = float(_base_formula_portions(request.present_today, 0))

    confidence = _calculate_confidence(
        request.school_id, 1, portions, request.present_today, request.total_students, has_model
    )
    return StandardPredictionResponse(predicted_portions=portions, confidence=confidence)


@app.post("/predict/multi", response_model=MultiDayPredictionResponse)
async def predict_multi(request: MultiDayPredictionRequest) -> MultiDayPredictionResponse:
    """Предсказание количества порций на несколько дней вперёд (до 5)."""
    predictions: List[DayPrediction] = []

    for i, day in enumerate(request.days):
        horizon = i + 1
        input_data = {
            "school_id": request.school_id,
            "age_group": request.age_group,
            "month": day.month,
            "day_of_month": day.day_of_month,
            "day_of_week": day.day_of_week,
            "is_pre_holiday": day.is_pre_holiday,
            "is_pre_vacation": day.is_pre_vacation,
            "is_holiday_calendar": day.is_holiday_calendar,
            "main_dish": day.main_dish,
            "total_students": request.total_students,
            "opt_outs": request.opt_outs,
            "present_today": request.present_today,
            "present_yesterday": request.present_yesterday,
            "present_t_minus_2": request.present_t_minus_2,
            "present_t_minus_3": request.present_t_minus_3,
            "present_avg_3d": request.present_avg_3d,
            "present_avg_7d": request.present_avg_7d,
            "present_trend": request.present_trend,
            "present_same_dow_last_week": request.present_same_dow_last_week,
            "temp_celsius": day.temp_celsius,
            "precipitation_mm": day.precipitation_mm,
            "is_rainy": day.is_rainy,
            "is_cold": day.is_cold,
        }
        portions = _standard_prediction(input_data, horizon=horizon)
        has_model = portions is not None
        if portions is None:
            portions = float(_base_formula_portions(request.present_today, 0))

        confidence = _calculate_confidence(
            request.school_id, horizon, portions, request.present_today, request.total_students, has_model
        )
        predictions.append(
            DayPrediction(
                date=day.date,
                predicted_portions=round(portions, 1),
                confidence=round(confidence, 2),
            )
        )

    return MultiDayPredictionResponse(
        school_id=request.school_id,
        predictions=predictions,
    )


@app.post("/predict/buffet", response_model=BuffetPredictionResponse)
async def predict_buffet(request: BuffetPredictionRequest) -> BuffetPredictionResponse:
    """Предсказание для шведского стола (1 день вперёд)."""
    input_data = {
        "school_id": request.school_id,
        "age_group": request.age_group,
        "month": request.month,
        "day_of_month": request.day_of_month,
        "day_of_week": request.day_of_week,
        "is_pre_holiday": request.is_pre_holiday,
        "is_pre_vacation": request.is_pre_vacation,
        "is_holiday_calendar": request.is_holiday_calendar,
        "main_dish": request.main_dish,
        "total_students": request.total_students,
        "opt_outs": request.opt_outs,
        "present_today": request.present_today,
        "present_yesterday": request.present_yesterday,
        "present_t_minus_2": request.present_t_minus_2,
        "present_t_minus_3": request.present_t_minus_3,
        "present_avg_3d": request.present_avg_3d,
        "present_avg_7d": request.present_avg_7d,
        "present_trend": request.present_trend,
        "present_same_dow_last_week": request.present_same_dow_last_week,
        "temp_celsius": request.temp_celsius,
        "precipitation_mm": request.precipitation_mm,
        "is_rainy": request.is_rainy,
        "is_cold": request.is_cold,
        "sold_hot_yesterday": request.sold_hot_yesterday,
        "sold_salad_yesterday": request.sold_salad_yesterday,
        "sold_bakery_yesterday": request.sold_bakery_yesterday,
    }
    result = _buffet_prediction(input_data, horizon=1)
    if result is None:
        # Fallback: базовый прогноз без буфера (буфер добавляет фронтенд при заявке)
        base = float(request.present_today)
        result = {
            cat: base * buffet_ratio_for(cat, request.age_group)
            for cat in BUFFET_CATEGORIES
        }
    return BuffetPredictionResponse(categories=result)


# ---------------------------------------------------------------------------
# Эндпоинт для интеграции с бэкендом (ML Service API)
# ---------------------------------------------------------------------------
@app.post("/predict", response_model=MLPredictResponse)
async def ml_predict(request: MLPredictRequest) -> MLPredictResponse:
    """
    ML Service API для интеграции с бэкендом.

    Получает полный расчёт прогноза (ForecastOut) и возвращает скорректированные
    поля: total_portions, confidence и прогноз на несколько дней.

    Логика:
    - meal_plan == "fixed" → стандартная модель (per-school или global)
    - meal_plan == "canteen" → буфетная модель (сумма горячего+салата+выпечки)
    """
    payload = request.model_dump()

    # Определяем целевую дату
    target_date = payload.get("date")
    if target_date is None:
        target_date = date.today()
    elif isinstance(target_date, str):
        target_date = date.fromisoformat(target_date)

    # Календарные признаки для целевой даты
    month = target_date.month
    day_of_month = target_date.day
    day_of_week = target_date.weekday()  # 0=Пн, 6=Вс
    holiday_calendar = get_holiday_calendar()
    is_pre_holiday = 1 if day_of_week == 4 else 0
    is_pre_vacation = 1 if (target_date + timedelta(days=1)).isoformat() in holiday_calendar else 0
    is_holiday_calendar_flag = 1 if target_date.isoformat() in holiday_calendar else 0

    school_id = payload.get("school_id", "")
    meal_plan = payload.get("meal_plan", "fixed")
    age_group = payload.get("age_group", "middle")
    total_students = payload.get("total_students", 0)
    opt_outs = payload.get("opt_outs", 0)
    present_today = payload.get("present", 0)

    # Исторические признаки
    present_yesterday = payload.get("present_yesterday") or max(0, present_today - 5)
    present_t_minus_2 = payload.get("present_t_minus_2") or max(0, present_today - 10)
    present_t_minus_3 = payload.get("present_t_minus_3") or max(0, present_today - 15)
    present_avg_3d = payload.get("present_avg_3d") or float(present_today)
    present_avg_7d = payload.get("present_avg_7d") or float(present_today)
    present_trend = payload.get("present_trend") or 0.0
    present_same_dow_last_week = payload.get("present_same_dow_last_week") or present_today

    # Погода
    weather = payload.get("weather") or {}
    main_dish = payload.get("main_dish")

    def _weather_for(target: date) -> dict:
        key = target.isoformat()
        w = weather.get(key, {}) if isinstance(weather, dict) else {}
        return {
            "temp_celsius": w.get("temp_celsius", 15.0),
            "precipitation_mm": w.get("precipitation_mm", 0.0),
            "is_rainy": w.get("is_rainy", 0),
            "is_cold": w.get("is_cold", 0),
        }

    buffer_percent = payload.get("buffer_percent", 8)

    # Основной прогноз (первый день / целевая дата)
    w_today = _weather_for(target_date)

    model_used = "fallback"
    total_portions: int | None = None
    confidence: float = 0.50

    buffet_categories: dict[str, float] | None = None

    if meal_plan == "canteen":
        # Буфетная модель
        buffet_result = _buffet_prediction({
            "school_id": school_id,
            "age_group": age_group,
            "month": month,
            "day_of_month": day_of_month,
            "day_of_week": day_of_week,
            "is_pre_holiday": is_pre_holiday,
            "is_pre_vacation": is_pre_vacation,
            "is_holiday_calendar": is_holiday_calendar_flag,
            "main_dish": main_dish,
            "total_students": total_students,
            "opt_outs": opt_outs,
            "present_today": present_today,
            "present_yesterday": present_yesterday,
            "present_t_minus_2": present_t_minus_2,
            "present_t_minus_3": present_t_minus_3,
            "present_avg_3d": present_avg_3d,
            "present_avg_7d": present_avg_7d,
            "present_trend": present_trend,
            "present_same_dow_last_week": present_same_dow_last_week,
            "temp_celsius": w_today["temp_celsius"],
            "precipitation_mm": w_today["precipitation_mm"],
            "is_rainy": w_today["is_rainy"],
            "is_cold": w_today["is_cold"],
            **{
                f"sold_{cat}_yesterday": payload.get("yesterday_total_qty", 0)
                * buffet_ratio_for(cat, age_group)
                for cat in BUFFET_CATEGORIES
            },
        }, horizon=1)

        if buffet_result is not None:
            buffet_categories = buffet_result
            total_portions = int(round(sum(buffet_result.values())))
            model_used = "ml"
            confidence = _calculate_confidence(
                school_id, 1, float(total_portions), present_today, total_students, True
            )
        else:
            # Fallback без буфера (буфер добавляет фронтенд)
            total_portions = int(present_today)
            confidence = 0.50
    else:
        # Стандартная модель
        portions = _standard_prediction({
            "school_id": school_id,
            "age_group": age_group,
            "month": month,
            "day_of_month": day_of_month,
            "day_of_week": day_of_week,
            "is_pre_holiday": is_pre_holiday,
            "is_pre_vacation": is_pre_vacation,
            "is_holiday_calendar": is_holiday_calendar_flag,
            "main_dish": main_dish,
            "total_students": total_students,
            "opt_outs": opt_outs,
            "present_today": present_today,
            "present_yesterday": present_yesterday,
            "present_t_minus_2": present_t_minus_2,
            "present_t_minus_3": present_t_minus_3,
            "present_avg_3d": present_avg_3d,
            "present_avg_7d": present_avg_7d,
            "present_trend": present_trend,
            "present_same_dow_last_week": present_same_dow_last_week,
            "temp_celsius": w_today["temp_celsius"],
            "precipitation_mm": w_today["precipitation_mm"],
            "is_rainy": w_today["is_rainy"],
            "is_cold": w_today["is_cold"],
        }, horizon=1)

        if portions is not None:
            total_portions = int(round(portions))
            model_used = "ml"
            confidence = _calculate_confidence(
                school_id, 1, float(total_portions), present_today, total_students, True
            )
        else:
            # Fallback без буфера (буфер добавляет фронтенд)
            total_portions = int(present_today)
            confidence = 0.50

    # Многодневный прогноз
    forecast_days_list: list[dict] = []
    forecast_days = int(payload.get("forecast_days", 5))
    for i in range(forecast_days):
        target = target_date + timedelta(days=i + 1)
        horizon = i + 1
        w = _weather_for(target)
        m = target.month
        d = target.day
        dw = target.weekday()
        # Календарные признаки для целевого дня
        is_pre_holiday_d = 1 if dw == 4 else 0
        is_holiday_calendar_d = 1 if target.isoformat() in holiday_calendar else 0
        is_pre_vacation_d = 1 if (target + timedelta(days=1)).isoformat() in holiday_calendar else 0

        if meal_plan == "canteen":
            result = _buffet_prediction({
                "school_id": school_id,
                "age_group": age_group,
                "month": m,
                "day_of_month": d,
                "day_of_week": dw,
                "is_pre_holiday": is_pre_holiday_d,
                "is_pre_vacation": is_pre_vacation_d,
                "is_holiday_calendar": is_holiday_calendar_d,
                "main_dish": main_dish,
                "total_students": total_students,
                "opt_outs": opt_outs,
                "present_today": present_today,
                "present_yesterday": present_yesterday,
                "present_t_minus_2": present_t_minus_2,
                "present_t_minus_3": present_t_minus_3,
                "present_avg_3d": present_avg_3d,
                "present_avg_7d": present_avg_7d,
                "present_trend": present_trend,
                "present_same_dow_last_week": present_same_dow_last_week,
                "temp_celsius": w["temp_celsius"],
                "precipitation_mm": w["precipitation_mm"],
                "is_rainy": w["is_rainy"],
                "is_cold": w["is_cold"],
                **{
                    f"sold_{cat}_yesterday": payload.get("yesterday_total_qty", 0)
                    * buffet_ratio_for(cat, age_group)
                    for cat in BUFFET_CATEGORIES
                },
            }, horizon=horizon)

            if result is not None:
                day_portions = int(round(sum(result.values())))
                day_conf = _calculate_confidence(
                    school_id, horizon, float(day_portions), present_today, total_students, True
                )
            else:
                # Fallback без буфера (буфер добавляет фронтенд)
                day_portions = int(present_today)
                day_conf = 0.50
        else:
            portions = _standard_prediction({
                "school_id": school_id,
                "age_group": age_group,
                "month": m,
                "day_of_month": d,
                "day_of_week": dw,
                "is_pre_holiday": is_pre_holiday_d,
                "is_pre_vacation": is_pre_vacation_d,
                "is_holiday_calendar": is_holiday_calendar_d,
                "main_dish": main_dish,
                "total_students": total_students,
                "opt_outs": opt_outs,
                "present_today": present_today,
                "present_yesterday": present_yesterday,
                "present_t_minus_2": present_t_minus_2,
                "present_t_minus_3": present_t_minus_3,
                "present_avg_3d": present_avg_3d,
                "present_avg_7d": present_avg_7d,
                "present_trend": present_trend,
                "present_same_dow_last_week": present_same_dow_last_week,
                "temp_celsius": w["temp_celsius"],
                "precipitation_mm": w["precipitation_mm"],
                "is_rainy": w["is_rainy"],
                "is_cold": w["is_cold"],
            }, horizon=horizon)

            if portions is not None:
                day_portions = int(round(portions))
                day_conf = _calculate_confidence(
                    school_id, horizon, portions, present_today, total_students, True
                )
            else:
                # Fallback без буфера (буфер добавляет фронтенд)
                day_portions = int(present_today)
                day_conf = 0.50

        forecast_days_list.append({
            "date": target.isoformat(),
            "horizon": horizon,
            "predicted_portions": day_portions,
            "confidence": round(day_conf, 2),
        })

    return MLPredictResponse(
        total_portions=total_portions,
        confidence=round(confidence, 2),
        forecast_days=forecast_days_list,
        model_used=model_used,
        categories=buffet_categories,
    )


# ---------------------------------------------------------------------------
# Запуск
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
