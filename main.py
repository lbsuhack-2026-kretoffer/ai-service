#!/usr/bin/env python3
"""
FastAPI backend сервис для предсказания школьного питания.

Сервис:
- Загружает модели при старте:
  * Глобальные модели по горизонтам: standard_global_model_h{h}.cbm (h=1..5)
    (fallback на standard_global_model.cbm, если горизонтной модели нет)
  * Дообученные модели школ: school_{id}_model_h{h}.cbm
  * Мультитаргетная модель шведского стола: buffet_model.cbm
- Эндпоинты:
  1. POST /predict/standard — стандартное питание на 1 день вперёд
  2. POST /predict/multi — стандартное питание на 5 дней вперёд
  3. POST /predict/buffet — шведский стол (мультитаргет)
  4. POST /predict — интеграция с бэкендом (ML Service API)

Новые признаки (точно такие же, как в train.py):
- Категориальные: school_id, age_group, month, day_of_month, day_of_week, main_dish
- Бинарные: is_pre_holiday, is_pre_vacation, is_rainy, is_cold
- Числовые: total_students, opt_outs, present_today, present_yesterday,
  present_avg_3d, present_avg_7d, present_trend, temp_celsius,
  precipitation_mm, dish_popularity
  (+ sold_hot/salad/bakery_yesterday для буфета)

Требования:
- Категориальные признаки передаются напрямую в CatBoost (без One-Hot Encoding)
- Используется pydantic для валидации входных данных
- Полностью готов к развертыванию через Docker
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

# ---------------------------------------------------------------------------
# Пути к моделям
# ---------------------------------------------------------------------------
MODELS_DIR = Path(__file__).parent / "models"

HORIZONS = [1, 2, 3, 4, 5]

STANDARD_GLOBAL_MODEL_PATHS = {
    h: MODELS_DIR / f"standard_global_model_h{h}.cbm" for h in HORIZONS
}
STANDARD_GLOBAL_MODEL_PATH_FALLBACK = MODELS_DIR / "standard_global_model.cbm"
BUFFET_MODEL_PATH = MODELS_DIR / "buffet_model.cbm"

# ---------------------------------------------------------------------------
# Списки фичей (строго в порядке обучения в train.py)
# ---------------------------------------------------------------------------
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

NUMERIC_FEATURES_STANDARD = [
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

NUMERIC_FEATURES_BUFFET = NUMERIC_FEATURES_STANDARD + [
    "sold_hot_yesterday",
    "sold_salad_yesterday",
    "sold_bakery_yesterday",
]

STANDARD_INPUT_FEATURES = CAT_FEATURES + BINARY_FEATURES + NUMERIC_FEATURES_STANDARD
BUFFET_INPUT_FEATURES = CAT_FEATURES + BINARY_FEATURES + NUMERIC_FEATURES_BUFFET

# ---------------------------------------------------------------------------
# Глобальные переменные моделей
# ---------------------------------------------------------------------------
standard_global_models: Dict[int, CatBoostRegressor] = {}  # horizon -> model
buffet_model: CatBoostRegressor | None = None
school_models: Dict[str, Dict[int, CatBoostRegressor]] = {}  # school_id -> {horizon: model}

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
    """Загрузка всех моделей при старте приложения."""
    global standard_global_models, buffet_model, school_models

    # Собираем модели локально, чтобы при неудаче не оставить globals в Broken состоянии
    new_global_models: Dict[int, CatBoostRegressor] = {}
    new_buffet: CatBoostRegressor | None = None
    new_school_models: Dict[str, Dict[int, CatBoostRegressor]] = {}

    # Глобальные модели по горизонтам (с fallback на старую h=1 модель)
    for h in HORIZONS:
        path = STANDARD_GLOBAL_MODEL_PATHS.get(h)
        if path is None:
            continue
        model = _load_single_model(path)
        if model is None:
            model = _load_single_model(STANDARD_GLOBAL_MODEL_PATH_FALLBACK)
        if model is not None:
            new_global_models[h] = model

    if not new_global_models:
        raise FileNotFoundError(
            f"Глобальные модели не найдены в {MODELS_DIR}. "
            f"Ожидались standard_global_model_h{{1..5}}.cbm или стандартный fallback"
        )

    # Буфетная модель
    if BUFFET_MODEL_PATH.exists():
        new_buffet = CatBoostRegressor()
        new_buffet.load_model(str(BUFFET_MODEL_PATH))

    # Дообученные модели школ по горизонтам
    pattern = re.compile(r"^(.+)_model_h(\d+)\.cbm$")
    excluded_prefixes = ("standard_global_model", "buffet_model", "standard_model")
    for model_file in MODELS_DIR.glob("*_model_h*.cbm"):
        name = model_file.name
        if any(name.startswith(p) for p in excluded_prefixes):
            continue
        match = pattern.match(name)
        if not match:
            continue
        school_id = match.group(1)
        horizon = int(match.group(2))
        model = CatBoostRegressor()
        model.load_model(str(model_file))
        new_school_models.setdefault(school_id, {})[horizon] = model

    # Атомарное присваивание globals
    standard_global_models = new_global_models
    buffet_model = new_buffet
    school_models = new_school_models

    logger.info(
        f"Модели загружены: горизонты {sorted(standard_global_models.keys())}, "
        f"школ {len(school_models)}, буфет={'да' if buffet_model else 'нет'}"
    )


# ---------------------------------------------------------------------------
# FastAPI
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Школьный Баланс-Пит - AI сервис предсказания питания",
    description="Сервис для предсказания количества порций питания (1 и 5 дней вперёд)",
    version="2.0.0",
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "horizons": sorted(standard_global_models.keys()),
        "schools": list(school_models.keys()),
        "buffet": buffet_model is not None,
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
    main_dish: Optional[str] = Field(None, description="Основное блюда дня (для fixed-питания)")
    temp_celsius: float = Field(..., description="Температура воздуха, °C")
    precipitation_mm: float = Field(..., ge=0.0, description="Осадки, мм")
    is_rainy: int = Field(..., ge=0, le=1, description="1 если дождь/снег")
    is_cold: int = Field(..., ge=0, le=1, description="1 если температура ниже порога")


class StandardPredictionRequest(BaseModel):
    """Схема запроса для предсказания стандартного питания (1 день)."""
    school_id: str = Field(..., description="Идентификатор школы")
    age_group: str = Field(..., description="Возрастная группа: 'junior', 'middle', 'senior'")
    month: int = Field(..., ge=1, le=12)
    day_of_month: int = Field(..., ge=1, le=31)
    day_of_week: int = Field(..., ge=0, le=6)
    is_pre_holiday: int = Field(..., ge=0, le=1)
    is_pre_vacation: int = Field(..., ge=0, le=1)
    main_dish: Optional[str] = Field(None, description="Основное блюда дня")
    total_students: int = Field(..., ge=1)
    opt_outs: int = Field(..., ge=0)
    present_today: int = Field(..., ge=1)
    present_yesterday: int = Field(..., ge=1)
    present_avg_3d: float = Field(..., ge=0.0, description="Средняя посещаемость за 3 дня")
    present_avg_7d: float = Field(..., ge=0.0, description="Средняя посещаемость за 7 дней")
    present_trend: float = Field(..., description="Тренд посещаемости (сегодня минус вчера)")
    temp_celsius: float = Field(..., description="Температура, °C")
    precipitation_mm: float = Field(..., ge=0.0, description="Осадки, мм")
    is_rainy: int = Field(..., ge=0, le=1)
    is_cold: int = Field(..., ge=0, le=1)

    @validator("age_group")
    def validate_age_group(cls, v):
        if v not in ["junior", "middle", "senior"]:
            raise ValueError("age_group должен быть 'junior', 'middle' или 'senior'")
        return v

    @validator("is_pre_holiday", "is_pre_vacation", "is_rainy", "is_cold")
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

    @validator("present_yesterday")
    def validate_present_yesterday(cls, v, values):
        total = values.get("total_students", 0)
        if v > total:
            raise ValueError("present_yesterday не может быть больше total_students")
        return v


class MultiDayPredictionRequest(BaseModel):
    """Схема запроса для предсказания на 5 дней вперёд."""
    school_id: str = Field(..., description="Идентификатор школы")
    age_group: str = Field(..., description="Возрастная группа")
    total_students: int = Field(..., ge=1)
    opt_outs: int = Field(..., ge=0)
    present_today: int = Field(..., ge=1)
    present_yesterday: int = Field(..., ge=1)
    present_avg_3d: float = Field(..., ge=0.0)
    present_avg_7d: float = Field(..., ge=0.0)
    present_trend: float = Field(...)
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

    @validator("present_yesterday")
    def validate_present_yesterday(cls, v, values):
        total = values.get("total_students", 0)
        if v > total:
            raise ValueError("present_yesterday не может быть больше total_students")
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
    main_dish: Optional[str] = None
    total_students: int = Field(..., ge=1)
    opt_outs: int = Field(..., ge=0)
    present_today: int = Field(..., ge=1)
    present_yesterday: int = Field(..., ge=1)
    present_avg_3d: float = Field(..., ge=0.0)
    present_avg_7d: float = Field(..., ge=0.0)
    present_trend: float = Field(...)
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

    @validator("is_pre_holiday", "is_pre_vacation", "is_rainy", "is_cold")
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

    @validator("present_yesterday")
    def validate_present_yesterday(cls, v, values):
        total = values.get("total_students", 0)
        if v > total:
            raise ValueError("present_yesterday не может быть больше total_students")
        return v


class BuffetPredictionResponse(BaseModel):
    hot: float
    salad: float
    bakery: float


# ---------------------------------------------------------------------------
# ML Service API (Backend Integration)
# ---------------------------------------------------------------------------
class MLPredictRequest(BaseModel):
    """
    Полоад от бэкенда (ForecastOut из backend/app/schemas.py).
    ml_client.py отправляет этот dict на /predict.
    """
    school_id: str
    school_name: str | None = None
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


class MLPredictResponse(BaseModel):
    """Ответ AI-сервиса — скорректированный прогноз."""
    total_portions: int | None = None
    confidence: float | None = None
    forecast_days: list[dict] | None = None  # [{date, predicted_portions, confidence}, ...]


# ---------------------------------------------------------------------------
# Хелперы
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


def _get_school_model(school_id: str, horizon: int) -> CatBoostRegressor:
    """Возвращает модель для школы и горизонта (с fallback на глобальную)."""
    school_horizon_models = school_models.get(school_id, {})
    if horizon in school_horizon_models:
        return school_horizon_models[horizon]
    if not standard_global_models:
        raise RuntimeError("Модель не загружена")
    return standard_global_models.get(horizon) or standard_global_models[1]


def _standard_prediction(request_data: dict, horizon: int = 1) -> float:
    """Внутренний предсказатель стандартного питания для заданного горизонта."""
    model = _get_school_model(request_data.get("school_id", ""), horizon)
    model_type = "school" if request_data.get("school_id", "") in school_models and horizon in school_models.get(request_data.get("school_id", ""), {}) else "global"

    pool = _make_prediction_pool(request_data, STANDARD_INPUT_FEATURES, CAT_FEATURES)
    prediction = model.predict(pool).flatten()[0]

    logger.info(
        f"Standard prediction h={horizon}: school={request_data.get('school_id')}, "
        f"model={model_type}, predicted_portions={prediction:.1f}"
    )
    return float(prediction)


def _buffet_prediction(request_data: dict) -> dict:
    """Внутренний предсказатель шведского стола."""
    if buffet_model is None:
        raise RuntimeError("Буфетная модель не загружена")

    pool = _make_prediction_pool(request_data, BUFFET_INPUT_FEATURES, CAT_FEATURES)
    prediction = buffet_model.predict(pool)

    if prediction.ndim != 2 or prediction.shape[1] != 3:
        raise RuntimeError(f"Ожидалось 3 предсказания, получено форма {prediction.shape}")

    return {
        "hot": float(prediction[0][0]),
        "salad": float(prediction[0][1]),
        "bakery": float(prediction[0][2]),
    }


def _calculate_confidence(
    school_id: str,
    horizon: int,
    prediction: float,
    present_today: int,
    total_students: int,
) -> float:
    """
    Расчёт уверенности в предсказании.
    
    Логика (балансируем между «недоедание» и «излишек»):
    - Базовая уверенность выше для школьной модели (больше данных о школе)
    - Если прогноз сильно отличается от посещаемости сегодня — ниже уверенность
    - Если школа маленькая — чуть выше неопределённость
    - Для далёких горизонтов (h=4,5) уверенность légèrement ниже
    """
    # Базовая уверенность по типу модели
    has_school_model = school_id in school_models and horizon in school_models.get(school_id, {})
    base_confidence = 0.88 if has_school_model else 0.80

    # Штраф за遠ний горизонт
    horizon_penalty = (horizon - 1) * 0.02

    # Штраф за大きное отклонение от сегодняшней посещаемости
    if present_today > 0:
        deviation = abs(prediction - present_today) / present_today
        deviation_penalty = min(deviation * 0.3, 0.15)
    else:
        deviation_penalty = 0.05

    # Штраф за маленькую школу (меньше статистики)
    size_penalty = 0.0 if total_students >= 200 else min((200 - total_students) / 200 * 0.05, 0.05)

    confidence = base_confidence - horizon_penalty - deviation_penalty - size_penalty
    return max(0.55, min(0.95, confidence))


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
        "main_dish": request.main_dish,
        "total_students": request.total_students,
        "opt_outs": request.opt_outs,
        "present_today": request.present_today,
        "present_yesterday": request.present_yesterday,
        "present_avg_3d": request.present_avg_3d,
        "present_avg_7d": request.present_avg_7d,
        "present_trend": request.present_trend,
        "temp_celsius": request.temp_celsius,
        "precipitation_mm": request.precipitation_mm,
        "is_rainy": request.is_rainy,
        "is_cold": request.is_cold,
    }
    portions = _standard_prediction(input_data, horizon=1)
    confidence = _calculate_confidence(
        request.school_id, 1, portions, request.present_today, request.total_students
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
            "main_dish": day.main_dish,
            "total_students": request.total_students,
            "opt_outs": request.opt_outs,
            "present_today": request.present_today,
            "present_yesterday": request.present_yesterday,
            "present_avg_3d": request.present_avg_3d,
            "present_avg_7d": request.present_avg_7d,
            "present_trend": request.present_trend,
            "temp_celsius": day.temp_celsius,
            "precipitation_mm": day.precipitation_mm,
            "is_rainy": day.is_rainy,
            "is_cold": day.is_cold,
        }
        portions = _standard_prediction(input_data, horizon=horizon)
        confidence = _calculate_confidence(
            request.school_id, horizon, portions, request.present_today, request.total_students
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
    """Предсказание для шведского стола."""
    input_data = {
        "school_id": request.school_id,
        "age_group": request.age_group,
        "month": request.month,
        "day_of_month": request.day_of_month,
        "day_of_week": request.day_of_week,
        "is_pre_holiday": request.is_pre_holiday,
        "is_pre_vacation": request.is_pre_vacation,
        "main_dish": request.main_dish,
        "total_students": request.total_students,
        "opt_outs": request.opt_outs,
        "present_today": request.present_today,
        "present_yesterday": request.present_yesterday,
        "present_avg_3d": request.present_avg_3d,
        "present_avg_7d": request.present_avg_7d,
        "present_trend": request.present_trend,
        "temp_celsius": request.temp_celsius,
        "precipitation_mm": request.precipitation_mm,
        "is_rainy": request.is_rainy,
        "is_cold": request.is_cold,
        "sold_hot_yesterday": request.sold_hot_yesterday,
        "sold_salad_yesterday": request.sold_salad_yesterday,
        "sold_bakery_yesterday": request.sold_bakery_yesterday,
    }
    result = _buffet_prediction(input_data)
    return BuffetPredictionResponse(
        hot=result["hot"], salad=result["salad"], bakery=result["bakery"]
    )


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
    is_pre_holiday = 1 if day_of_week == 4 else 0
    is_pre_vacation = 0

    school_id = payload.get("school_id", "")
    meal_plan = payload.get("meal_plan", "fixed")
    total_students = payload.get("total_students", 0)
    opt_outs = payload.get("opt_outs", 0)
    present_today = payload.get("present", 0)
    present_yesterday = max(0, present_today - 5)  # приблизительно

    # Исторические признаки
    present_avg_3d = payload.get("present_avg_3d") or float(present_today)
    present_avg_7d = payload.get("present_avg_7d") or float(present_today)
    present_trend = payload.get("present_trend") or 0.0

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

    # Основной прогноз (первый день / целевая дата)
    w_today = _weather_for(target_date)

    if meal_plan == "canteen":
        # Буфетная модель
        buffet_result = _buffet_prediction({
            "school_id": school_id,
            "age_group": "middle",
            "month": month,
            "day_of_month": day_of_month,
            "day_of_week": day_of_week,
            "is_pre_holiday": is_pre_holiday,
            "is_pre_vacation": is_pre_vacation,
            "main_dish": main_dish,
            "total_students": total_students,
            "opt_outs": opt_outs,
            "present_today": present_today,
            "present_yesterday": present_yesterday,
            "present_avg_3d": present_avg_3d,
            "present_avg_7d": present_avg_7d,
            "present_trend": present_trend,
            "temp_celsius": w_today["temp_celsius"],
            "precipitation_mm": w_today["precipitation_mm"],
            "is_rainy": w_today["is_rainy"],
            "is_cold": w_today["is_cold"],
            "sold_hot_yesterday": payload.get("yesterday_total_qty", 0) * 0.42,
            "sold_salad_yesterday": payload.get("yesterday_total_qty", 0) * 0.18,
            "sold_bakery_yesterday": payload.get("yesterday_total_qty", 0) * 0.22,
        })
        total_portions = int(round(buffet_result["hot"] + buffet_result["salad"] + buffet_result["bakery"]))
        confidence = 0.85
    else:
        # Стандартная модель
        total_portions = int(round(_standard_prediction({
            "school_id": school_id,
            "age_group": "middle",
            "month": month,
            "day_of_month": day_of_month,
            "day_of_week": day_of_week,
            "is_pre_holiday": is_pre_holiday,
            "is_pre_vacation": is_pre_vacation,
            "main_dish": main_dish,
            "total_students": total_students,
            "opt_outs": opt_outs,
            "present_today": present_today,
            "present_yesterday": present_yesterday,
            "present_avg_3d": present_avg_3d,
            "present_avg_7d": present_avg_7d,
            "present_trend": present_trend,
            "temp_celsius": w_today["temp_celsius"],
            "precipitation_mm": w_today["precipitation_mm"],
            "is_rainy": w_today["is_rainy"],
            "is_cold": w_today["is_cold"],
        }, horizon=1)))
        confidence = _calculate_confidence(
            school_id, 1, float(total_portions), present_today, total_students
        )

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
        pre_h = 1 if dw == 4 else 0

        portions = _standard_prediction({
            "school_id": school_id,
            "age_group": "middle",
            "month": m,
            "day_of_month": d,
            "day_of_week": dw,
            "is_pre_holiday": pre_h,
            "is_pre_vacation": 0,
            "main_dish": main_dish,
            "total_students": total_students,
            "opt_outs": opt_outs,
            "present_today": present_today,
            "present_yesterday": present_yesterday,
            "present_avg_3d": present_avg_3d,
            "present_avg_7d": present_avg_7d,
            "present_trend": present_trend,
            "temp_celsius": w["temp_celsius"],
            "precipitation_mm": w["precipitation_mm"],
            "is_rainy": w["is_rainy"],
            "is_cold": w["is_cold"],
        }, horizon=horizon)

        day_conf = _calculate_confidence(
            school_id, horizon, portions, present_today, total_students
        )
        forecast_days_list.append({
            "date": target.isoformat(),
            "horizon": horizon,
            "predicted_portions": int(round(portions)),
            "confidence": round(day_conf, 2),
        })

    return MLPredictResponse(
        total_portions=total_portions,
        confidence=round(confidence, 2),
        forecast_days=forecast_days_list,
    )


# ---------------------------------------------------------------------------
# Запуск
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
