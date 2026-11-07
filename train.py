#!/usr/bin/env python3
"""
Скрипт генерации данных и обучения моделей CatBoost
для системы предиктивного заказа школьного питания (версия 2.0 — многогоризонт).

Источник данных: PostgreSQL база бэкенда (таблицы schools, classes, students, attendance, canteen_sales).

Модели:
1. StandardModel — предсказание общего количества порций на хORIZONT днях вперёд (h=1..5)
   - Глобальная модель по каждому горизонту (standard_global_model_h{h}.cbm)
   - Дообученные модели для каждой школы (school_{id}_model_h{h}.cbm через Transfer Learning)
2. BuffetModel — мультитаргетная модель для шведского стола (MultiRMSE)

Новые признаки (v2.0):
- main_dish — основное блюда дня (категориальный)
- is_rainy, is_cold — погодные бинарные признаки
- temp_celsius, precipitation_mm — погодные числовые признаки
- dish_popularity — история популярности блюда (0..1)
- present_avg_3d, present_avg_7d — скользящее среднее посещаемости
- present_trend — тренд посещаемости (сегодня минус вчера)

Категориальные признаки передаются в CatBoost напрямую (без One-Hot Encoding).
"""

from __future__ import annotations

import logging
import os
import warnings
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine, text

warnings.filterwarnings("ignore", category=UserWarning, module="catboost")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Пути к моделям
MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

HORIZONS = [1, 2, 3, 4, 5]

STANDARD_GLOBAL_MODEL_PATHS = {
    h: MODELS_DIR / f"standard_global_model_h{h}.cbm" for h in HORIZONS
}
BUFFET_MODEL_PATH = MODELS_DIR / "buffet_model.cbm"

# URL подключения к PostgreSQL бэкенда
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://school:school@db:5432/school_balance",
)

# Список возрастных групп бэкенда
AGE_GROUPS = ["junior", "middle", "senior"]

# Категориальные признаки (строго в порядке обучения — должны совпадать с main.py)
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

TARGET_STANDARD = "target_portions_tomorrow"  # базовый таргет (h=1)
TARGETS_BUFFET = [
    "target_hot_tomorrow",
    "target_salad_tomorrow",
    "target_bakery_tomorrow",
]

# Популярные школы для дообучения (все из БД)
TOP_SCHOOLS_COUNT = 10

# Порог холодной температуры (°C) для признака is_cold
COLD_THRESHOLD = 10.0

# Популярные блюда для синтетики
MAIN_DISHES = [
    "гречка с котлетой",
    "макароны по-флотски",
    "пюре с отбивной",
    "берёзовая каша",
    "рыба запечённая",
    "куриный суп",
    "борщ",
    "oven-пицца",
]
DISH_POPULARITY = {
    "гречка с котлетой": 0.85,
    "макароны по-флотски": 0.78,
    "пюре с отбивной": 0.82,
    "берёзовая каша": 0.45,
    "рыба запечённая": 0.60,
    "куриный суп": 0.70,
    "борщ": 0.68,
    "oven-пицца": 0.90,
}


def set_seed(seed: int = 42) -> None:
    """Фиксация случайности для воспроизводимости."""
    np.random.seed(seed)


def load_backend_data(database_url: str | None = None) -> pd.DataFrame:
    """
    Загружает данные из БД бэкенда (PostgreSQL) и строит обучающую выборку.

    Для каждого дня с посещаемостью формируем запись:
    - school_id, age_group, month, day_of_month, day_of_week, main_dish
    - is_pre_holiday, is_pre_vacation, is_rainy, is_cold
    - total_students, opt_outs, present_today, present_yesterday
    - present_avg_3d, present_avg_7d, present_trend
    - temp_celsius, precipitation_mm
    - для canteen-школ: sold_hot/salad/bakery_yesterday
    - таргеты по каждому горизонту h=1..5: present_today через h дней
    """
    url = database_url or DATABASE_URL
    logger.info(f"Загрузка данных из PostgreSQL: {url}")

    try:
        engine = create_engine(url, pool_pre_ping=True)

        schools_df = pd.read_sql("SELECT * FROM schools", engine)
        logger.info(f"Школ: {len(schools_df)}")
        print(schools_df[["id", "name", "meal_plan"]].to_string(index=False))

        classes_df = pd.read_sql("SELECT * FROM classes", engine)
        logger.info(f"Классов: {len(classes_df)}")

        students_df = pd.read_sql("SELECT * FROM students", engine)
        logger.info(f"Учеников: {len(students_df)}")

        att_df = pd.read_sql("SELECT * FROM attendance", engine)
        att_df["date"] = pd.to_datetime(att_df["date"]).dt.date
        logger.info(f"Записей посещаемости: {len(att_df)}")

        sales_df = pd.read_sql("SELECT * FROM canteen_sales", engine)
        sales_df["date"] = pd.to_datetime(sales_df["date"]).dt.date
        logger.info(f"Записей продаж: {len(sales_df)}")

        # Предварительно считаем present для каждого (class_id, date)
        present_lookup: dict[tuple[str, date], int] = {}
        for _, att in att_df.iterrows():
            class_id = att["class_id"]
            att_date = att["date"]

            cls_row = classes_df[classes_df["id"] == class_id]
            if cls_row.empty:
                continue
            school_id = cls_row.iloc[0]["school_id"]

            class_students = students_df[students_df["class_id"] == class_id]
            total_students = len(class_students)
            opt_outs = len(class_students[class_students["meal_status"] == "optOut"])
            absent_ids = set(att.get("absent_student_ids", []) or [])
            regular_ids = set(class_students[class_students["meal_status"] == "regular"]["id"])
            absent = len(absent_ids & regular_ids)
            present_today = max(0, total_students - opt_outs - absent)
            present_lookup[(class_id, att_date)] = present_today

        # Для каждого класса и даты посещаемости формируем запись
        records = []
        for _, att in att_df.iterrows():
            class_id = att["class_id"]
            att_date = att["date"]

            cls_row = classes_df[classes_df["id"] == class_id]
            if cls_row.empty:
                continue
            school_id = cls_row.iloc[0]["school_id"]
            age_group = cls_row.iloc[0]["age_group"]

            school_row = schools_df[schools_df["id"] == school_id]
            if school_row.empty:
                continue
            meal_plan = school_row.iloc[0]["meal_plan"]

            class_students = students_df[students_df["class_id"] == class_id]
            total_students = len(class_students)
            opt_outs = len(class_students[class_students["meal_status"] == "optOut"])

            absent_ids = set(att.get("absent_student_ids", []) or [])
            regular_ids = set(class_students[class_students["meal_status"] == "regular"]["id"])
            absent = len(absent_ids & regular_ids)

            present_today = max(0, total_students - opt_outs - absent)

            # Present вчера для этого же класса
            yesterday = att_date - timedelta(days=1)
            present_yesterday = present_lookup.get((class_id, yesterday), present_today)

            # Скользящее среднее за 3 и 7 дней (включая сегодня)
            window_3 = []
            window_7 = []
            for d_offset in range(3):
                d = att_date - timedelta(days=d_offset)
                window_3.append(present_lookup.get((class_id, d), present_today))
            for d_offset in range(7):
                d = att_date - timedelta(days=d_offset)
                window_7.append(present_lookup.get((class_id, d), present_today))
            present_avg_3d = float(np.mean(window_3))
            present_avg_7d = float(np.mean(window_7))
            present_trend = float(present_today - present_yesterday)

            # Календарные признаки
            month = att_date.month
            day_of_month = att_date.day
            day_of_week = att_date.weekday()  # 0=Пн, 6=Вс
            is_pre_holiday = 1 if day_of_week == 4 else 0
            is_pre_vacation = 0

            # Погодные признаки (в БД нет — заполняем нулями/нейтральными значениями,
            # реальная погода будет подтягиваться бэкендом при инференсе)
            temp_celsius = 15.0
            precipitation_mm = 0.0
            is_rainy = 0
            is_cold = 0 if temp_celsius >= COLD_THRESHOLD else 1

            # Блюдо дня (в БД нет таблицы меню для fixed-школ — будет подтягиваться бэкендом)
            main_dish = None
            dish_popularity = 0.5

            record = {
                "school_id": school_id,
                "class_id": class_id,
                "age_group": age_group,
                "month": month,
                "day_of_month": day_of_month,
                "day_of_week": day_of_week,
                "is_pre_holiday": is_pre_holiday,
                "is_pre_vacation": is_pre_vacation,
                "main_dish": main_dish,
                "dish_popularity": dish_popularity,
                "total_students": total_students,
                "opt_outs": opt_outs,
                "present_today": present_today,
                "present_yesterday": present_yesterday,
                "present_avg_3d": present_avg_3d,
                "present_avg_7d": present_avg_7d,
                "present_trend": present_trend,
                "temp_celsius": temp_celsius,
                "precipitation_mm": precipitation_mm,
                "is_rainy": is_rainy,
                "is_cold": is_cold,
                "meal_plan": meal_plan,
                "date": att_date,
            }

            # Для canteen-школ добавляем продажи вчера
            if meal_plan == "canteen":
                y_date = att_date - timedelta(days=1)
                cat_map = {"cat-hot": "sold_hot_yesterday", "cat-salad": "sold_salad_yesterday", "cat-bakery": "sold_bakery_yesterday"}
                for cat_id, col in cat_map.items():
                    sale = sales_df[
                        (sales_df["school_id"] == school_id)
                        & (sales_df["date"] == y_date)
                        & (sales_df["category_id"] == cat_id)
                    ]
                    record[col] = int(sale["qty"].iloc[0]) if not sale.empty else 0

            records.append(record)

        df = pd.DataFrame(records)
        if df.empty:
            logger.warning("БД пуста — генерируем синтетические данные как fallback")
            return _generate_synthetic_fallback()
    except Exception as e:
        logger.warning(f"Не удалось загрузить данные из PostgreSQL ({e}) — переходим на синтетический fallback")
        return _generate_synthetic_fallback()

    # Таргеты по каждому горизонту h=1..5: present_today через h дней для того же класса
    df = df.sort_values(["school_id", "class_id", "date"])
    for h in HORIZONS:
        df[f"target_portions_h{h}"] = df.groupby(["school_id", "class_id"])["present_today"].shift(-h)
        # Для последних дней используем present_today как fallback
        df[f"target_portions_h{h}"] = df[f"target_portions_h{h}"].fillna(df["present_today"])

    # Базовый таргет (h=1) для обратной совместимости
    df["target_portions_tomorrow"] = df["target_portions_h1"]

    # Буфетные таргеты (для canteen-школ)
    buffet_mask = df["meal_plan"] == "canteen"
    if buffet_mask.any():
        df.loc[buffet_mask, "target_hot_tomorrow"] = (
            df.loc[buffet_mask, "target_portions_h1"] * 0.42
        ).astype(int)
        df.loc[buffet_mask, "target_salad_tomorrow"] = (
            df.loc[buffet_mask, "target_portions_h1"] * 0.18
        ).astype(int)
        df.loc[buffet_mask, "target_bakery_tomorrow"] = (
            df.loc[buffet_mask, "target_portions_h1"] * 0.22
        ).astype(int)

    logger.info(f"Построено записей для обучения: {len(df)}")
    return df


def _generate_synthetic_fallback() -> pd.DataFrame:
    """Генерация синтетических данных с новыми признаками (погода, блюдо, история)."""
    logger.info("Генерация синтетических данных (fallback, v2.0)...")
    np.random.seed(42)

    schools = [
        {"school_id": "school-a", "name": "Гимназия №12", "meal_plan": "fixed", "baseline": 0.95, "city": "Минск", "address": "ул. Кальварийская, 12"},
        {"school_id": "school-b", "name": "Средняя школа №34", "meal_plan": "canteen", "baseline": 1.05, "city": "Минск", "address": "пр-т Независимости, 34"},
        {"school_id": "school-c", "name": "Лицей №7", "meal_plan": "fixed", "baseline": 0.90, "city": "Минск", "address": "пр-т Победителей, 7"},
        {"school_id": "school-d", "name": "Школа №101", "meal_plan": "canteen", "baseline": 1.10, "city": "Минск", "address": "ул. Независимости, 101"},
        {"school_id": "school-e", "name": "Гимназия №5", "meal_plan": "fixed", "baseline": 0.85, "city": "Минск", "address": "ул. Ленина, 5"},
        {"school_id": "school-f", "name": "Школа №22", "meal_plan": "canteen", "baseline": 1.00, "city": "Минск", "address": "ул. Каспийская, 22"},
    ]

    classes_per_school = {
        "school-a": [("1A", "junior", 25), ("1B", "junior", 24), ("5A", "middle", 28), ("5B", "middle", 27), ("9A", "senior", 26), ("9B", "senior", 25)],
        "school-b": [("2A", "junior", 30), ("2B", "junior", 29), ("6A", "middle", 31), ("6B", "middle", 30), ("10A", "senior", 29), ("10B", "senior", 28)],
        "school-c": [("3A", "junior", 22), ("3B", "junior", 23), ("7A", "middle", 25), ("7B", "middle", 24), ("11A", "senior", 23), ("11B", "senior", 22)],
        "school-d": [("4A", "junior", 28), ("4B", "junior", 27), ("8A", "middle", 29), ("8B", "middle", 28), ("10A", "senior", 27), ("10B", "senior", 26)],
        "school-e": [("1A", "junior", 20), ("1B", "junior", 21), ("5A", "middle", 23), ("5B", "middle", 22), ("9A", "senior", 21), ("9B", "senior", 20)],
        "school-f": [("2A", "junior", 26), ("2B", "junior", 25), ("6A", "middle", 27), ("6B", "middle", 26), ("10A", "senior", 25), ("10B", "senior", 24)],
    }

    start_date = date(2026, 9, 1)
    end_date = date(2026, 12, 18)
    all_dates = []
    d = start_date
    while d <= end_date:
        if d.weekday() < 5:
            all_dates.append(d)
        d += timedelta(days=1)

    pre_holiday_dates = {date(2026, 10, 30), date(2026, 12, 18)}
    vacation_dates = {date(2026, 10, 31), date(2026, 11, 1), date(2026, 12, 19), date(2026, 12, 20)}

    month_factor = {9: 1.05, 10: 1.02, 11: 1.00, 12: 0.92}
    dow_factor = {0: 1.02, 1: 1.00, 2: 0.98, 3: 0.95, 4: 0.85}
    age_factor_target = {"junior": 0.90, "middle": 1.00, "senior": 1.10}

    age_hot = {"junior": 0.55, "middle": 0.60, "senior": 0.50}
    age_salad = {"junior": 0.10, "middle": 0.20, "senior": 0.30}
    age_bakery = {"junior": 0.35, "middle": 0.25, "senior": 0.20}

    # Погода по месяцам (средняя температура, вероятность дождя)
    month_weather = {
        9: {"temp": 14.0, "rain_prob": 0.30},
        10: {"temp": 9.0, "rain_prob": 0.40},
        11: {"temp": 3.0, "rain_prob": 0.50},
        12: {"temp": -2.0, "rain_prob": 0.35},
    }

    records = []
    for school in schools:
        sid = school["school_id"]
        meal_plan = school["meal_plan"]
        baseline = school["baseline"]

        for class_id, age_group, class_size in classes_per_school[sid]:
            opt_out_ratio = np.random.uniform(0.05, 0.15)
            opt_outs_base = max(1, int(class_size * opt_out_ratio))

            # Генерируем последовательность посещаемости с автокорреляцией
            present_history = []
            for att_date in all_dates:
                dow = att_date.weekday()
                month = att_date.month

                is_pre_holiday = 1 if (att_date in pre_holiday_dates or dow == 4) else 0
                is_pre_vacation = 1 if att_date in pre_holiday_dates else 0

                total_students = max(10, class_size + np.random.randint(-2, 3))
                opt_outs = max(0, min(opt_outs_base + np.random.randint(-1, 2), total_students))

                # Погода
                mw = month_weather.get(month, {"temp": 10.0, "rain_prob": 0.35})
                temp_celsius = float(np.random.normal(mw["temp"], 3.0))
                precipitation_mm = float(np.random.exponential(2.0) if np.random.random() < mw["rain_prob"] else 0.0)
                is_rainy = 1 if precipitation_mm > 0.5 else 0
                is_cold = 1 if temp_celsius < COLD_THRESHOLD else 0

                # Блюдо дня
                main_dish = np.random.choice(MAIN_DISHES)

                # Базовое количество присутствующих (с погодным влиянием)
                weather_factor = 1.0 - 0.05 * is_rainy - 0.03 * is_cold

                base_present = (
                    (total_students - opt_outs)
                    * baseline
                    * month_factor.get(month, 1.0)
                    * dow_factor.get(dow, 1.0)
                    * age_factor_target[age_group]
                    * weather_factor
                )
                present_today = int(np.random.normal(base_present, base_present * 0.05))
                present_today = max(0, min(present_today, total_students - opt_outs))

                present_history.append(present_today)

                records.append({
                    "school_id": sid,
                    "class_id": class_id,
                    "age_group": age_group,
                    "month": month,
                    "day_of_month": att_date.day,
                    "day_of_week": dow,
                    "is_pre_holiday": is_pre_holiday,
                    "is_pre_vacation": is_pre_vacation,
                    "main_dish": main_dish,
                    "total_students": total_students,
                    "opt_outs": opt_outs,
                    "present_today": present_today,
                    "present_yesterday": present_history[-2] if len(present_history) >= 2 else present_today,
                    "meal_plan": meal_plan,
                    "date": att_date,
                    "temp_celsius": temp_celsius,
                    "precipitation_mm": precipitation_mm,
                    "is_rainy": is_rainy,
                    "is_cold": is_cold,
                })

    df = pd.DataFrame(records)
    if df.empty:
        logger.warning("Синтетические данные не сгенерированы")
        return df

    df = df.sort_values(["school_id", "class_id", "date"])

    # present_avg_3d, present_avg_7d, present_trend
    for h in (3, 7):
        df[f"present_avg_{h}d"] = df.groupby(["school_id", "class_id"])["present_today"].transform(
            lambda s: s.rolling(window=h, min_periods=1).mean()
        )
    df["present_trend"] = df.groupby(["school_id", "class_id"])["present_today"].diff(1).fillna(0.0)

    # present_yesterday заполняем через shift (уже частично заполнено выше, но пересчитаем корректно)
    df["present_yesterday"] = df.groupby(["school_id", "class_id"])["present_today"].shift(1)
    first_mask = df["present_yesterday"].isna()
    df.loc[first_mask, "present_yesterday"] = df.loc[first_mask, "present_today"]

    # Таргеты по горизонтам
    for h in HORIZONS:
        df[f"target_portions_h{h}"] = df.groupby(["school_id", "class_id"])["present_today"].shift(-h)
        df[f"target_portions_h{h}"] = df[f"target_portions_h{h}"].fillna(df["present_today"])
    df["target_portions_tomorrow"] = df["target_portions_h1"]

    # Буфетные фичи и таргеты для canteen-школ
    buffet_mask = df["meal_plan"] == "canteen"
    if buffet_mask.any():
        df.loc[buffet_mask, "sold_hot_yesterday"] = (
            df.loc[buffet_mask, "present_yesterday"]
            * df.loc[buffet_mask, "age_group"].map(age_hot)
            * np.random.uniform(0.85, 1.15, size=int(buffet_mask.sum()))
        ).clip(lower=0).astype(int)

        df.loc[buffet_mask, "sold_salad_yesterday"] = (
            df.loc[buffet_mask, "present_yesterday"]
            * df.loc[buffet_mask, "age_group"].map(age_salad)
            * np.random.uniform(0.85, 1.15, size=int(buffet_mask.sum()))
        ).clip(lower=0).astype(int)

        df.loc[buffet_mask, "sold_bakery_yesterday"] = (
            df.loc[buffet_mask, "present_yesterday"]
            * df.loc[buffet_mask, "age_group"].map(age_bakery)
            * np.random.uniform(0.85, 1.15, size=int(buffet_mask.sum()))
        ).clip(lower=0).astype(int)

        df.loc[buffet_mask, "target_hot_tomorrow"] = (
            df.loc[buffet_mask, "target_portions_h1"]
            * df.loc[buffet_mask, "age_group"].map(age_hot)
        ).astype(int)
        df.loc[buffet_mask, "target_salad_tomorrow"] = (
            df.loc[buffet_mask, "target_portions_h1"]
            * df.loc[buffet_mask, "age_group"].map(age_salad)
        ).astype(int)
        df.loc[buffet_mask, "target_bakery_tomorrow"] = (
            df.loc[buffet_mask, "target_portions_h1"]
            * df.loc[buffet_mask, "age_group"].map(age_bakery)
        ).astype(int)

    logger.info(f"Построено синтетических записей: {len(df)}")
    return df


def _make_pool(
    df: pd.DataFrame,
    cat_feature_names: list[str],
    label: pd.Series | np.ndarray | None = None,
) -> Pool:
    """Создает CatBoost Pool с категориальными признаками."""
    dfc = df.copy()
    for c in cat_feature_names:
        if c in dfc.columns:
            dfc[c] = dfc[c].astype(str)
    cat_positions = [list(dfc.columns).index(c) for c in cat_feature_names]
    return Pool(dfc, label=label, cat_features=cat_positions)


def _fit_global_model(
    X_train, y_train, X_test, y_test, horizon: int
) -> CatBoostRegressor:
    """Обучает глобальную модель CatBoostRegressor для заданного горизонта."""
    logger.info(f"Обучение глобальной модели (h={horizon}, RMSE)...")

    train_pool = _make_pool(X_train, CAT_FEATURES, label=y_train)
    test_pool = _make_pool(X_test, CAT_FEATURES, label=y_test)

    model = CatBoostRegressor(
        iterations=600,
        learning_rate=0.05,
        depth=6,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=42,
        verbose=100,
        early_stopping_rounds=50,
        allow_writing_files=False,
    )
    model.fit(train_pool, eval_set=test_pool, use_best_model=True)

    logger.info(
        f"Глобальная модель h={horizon}: best_iteration={model.best_iteration_}, "
        f"best_score={model.best_score_['validation']['RMSE']:.4f}"
    )
    return model


def _fit_school_model(
    school_id: str, X_school: pd.DataFrame, y_school: pd.Series,
    global_model: CatBoostRegressor, horizon: int,
) -> CatBoostRegressor | None:
    """Дообучение модели для конкретной школы через Transfer Learning."""
    if len(X_school) < 10:
        logger.warning(f"Для {school_id} h={horizon} мало данных ({len(X_school)}), пропускаем")
        return None

    X_s_train, X_s_test, y_s_train, y_s_test = train_test_split(
        X_school, y_school, test_size=0.2, random_state=42
    )

    train_pool = _make_pool(X_s_train, CAT_FEATURES, label=y_s_train)
    test_pool = _make_pool(X_s_test, CAT_FEATURES, label=y_s_test)

    fine_tuned = CatBoostRegressor(
        iterations=300,
        learning_rate=0.03,
        depth=6,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=42,
        verbose=50,
        early_stopping_rounds=30,
        allow_writing_files=False,
    )
    fine_tuned.fit(
        train_pool,
        eval_set=test_pool,
        init_model=global_model,
        use_best_model=True,
    )

    logger.info(
        f"{school_id} h={horizon}: best_iteration={fine_tuned.best_iteration_}, "
        f"best_score={fine_tuned.best_score_['validation']['RMSE']:.4f}"
    )
    return fine_tuned


def train_standard_models(df: pd.DataFrame) -> tuple:
    """
    Обучение модели стандартного питания по всем горизонтам h=1..5.
    Возвращает (dict horizon->global_model, dict school_id->dict horizon->model).
    """
    logger.info("=" * 60)
    logger.info("ОБУЧЕНИЕ МОДЕЛИ 'СТАНДАРТНОЕ ПИТАНИЕ' (многогоризонт, h=1..5)")
    logger.info("=" * 60)

    global_models: dict[int, CatBoostRegressor] = {}
    school_models_all: dict[str, dict[int, CatBoostRegressor]] = {}

    for h in HORIZONS:
        target_col = f"target_portions_h{h}"
        logger.info(f"\n--- Горизонт h={h} (таргет: {target_col}) ---")

        X = df[STANDARD_INPUT_FEATURES]
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=X["school_id"]
        )

        # 1. Глобальная модель для этого горизонта
        global_model = _fit_global_model(X_train, y_train, X_test, y_test, h)
        global_models[h] = global_model

        # Сохранение глобальной модели
        global_model.save_model(str(STANDARD_GLOBAL_MODEL_PATHS[h]))
        logger.info(f"Глобальная модель h={h}: {STANDARD_GLOBAL_MODEL_PATHS[h]}")

        # 2. Дообучение для топ-школь
        school_counts = X["school_id"].value_counts()
        top_school_ids = school_counts.head(TOP_SCHOOLS_COUNT).index.tolist()
        logger.info(f"Школы для дообучения h={h}: {top_school_ids}")

        school_models_h: dict[int, CatBoostRegressor] = {}
        for school_id in top_school_ids:
            mask = X["school_id"] == school_id
            model = _fit_school_model(school_id, X[mask], y[mask], global_model, h)
            if model is not None:
                school_models_h[school_id] = model

        # Сохранение школьных моделей
        for school_id, model in school_models_h.items():
            model_path = MODELS_DIR / f"{school_id}_model_h{h}.cbm"
            model.save_model(str(model_path))
            logger.info(f"Модель для {school_id} h={h}: {model_path}")

        # Аккумулируем школьные модели
        for school_id, model in school_models_h.items():
            school_models_all.setdefault(school_id, {})[h] = model

    return global_models, school_models_all


def train_buffet_model(df: pd.DataFrame) -> CatBoostRegressor:
    """Обучение мультитаргетной модели для шведского стола."""
    logger.info("=" * 60)
    logger.info("ОБУЧЕНИЕ МОДЕЛИ 'ШВЕДСКИЙ СТОЛ' (Multi-target)")
    logger.info("=" * 60)

    buffet_df = df[df["meal_plan"] == "canteen"].copy()
    if buffet_df.empty:
        logger.warning("Нет canteen-данных — буфетная модель не обучена")
        return None

    X = buffet_df[BUFFET_INPUT_FEATURES]
    y = buffet_df[TARGETS_BUFFET]

    train_pool = _make_pool(X, CAT_FEATURES, label=y.to_numpy())

    model = CatBoostRegressor(
        iterations=600,
        learning_rate=0.05,
        depth=6,
        loss_function="MultiRMSE",
        eval_metric="MultiRMSE",
        random_seed=42,
        verbose=100,
        early_stopping_rounds=50,
        allow_writing_files=False,
    )
    model.fit(train_pool, use_best_model=True)

    logger.info(f"Буфетная модель: best_iteration={model.best_iteration_}")
    model.save_model(str(BUFFET_MODEL_PATH))
    logger.info(f"Буфетная модель: {BUFFET_MODEL_PATH}")
    return model


def main() -> None:
    logger.info("Запуск пайплайна обучения моделей 'Школьный Баланс-Пит' v2.0")
    set_seed(42)

    df = load_backend_data()

    # 1. Стандартное питание (многогоризонт)
    global_models, school_models_all = train_standard_models(df)

    # 2. Шведский стол
    buffet_model = train_buffet_model(df)

    logger.info("=" * 60)
    logger.info("ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО")
    logger.info(f"Горизонты: {HORIZONS}")
    logger.info(f"Модели: {MODELS_DIR}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
