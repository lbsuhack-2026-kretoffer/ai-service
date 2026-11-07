#!/usr/bin/env python3
"""
Скрипт генерации данных и обучения моделей CatBoost
для системы предиктивного заказа школьного питания (версия 2.1 — многогоризонт + буфет).

Источник данных: PostgreSQL база бэкенда (таблицы schools, classes, students, attendance, canteen_sales).

Модели:
1. StandardModel — предсказание общего количества порций на hORIZONT днях вперёд (h=1..5)
   - Глобальная модель по каждому горизонту (standard_global_model_v2_h{h}.cbm)
   - Дообученные модели для каждой школы (school_{id}_model_v2_h{h}.cbm через Transfer Learning)
2. BuffetModel — мультитаргетная модель для шведского стола на hORIZONT днях вперёд (h=1..5)
   - Глобальная модель по каждому горизонту (buffet_model_v2_h{h}.cbm)
   - Дообученные модели для каждой школы

Новые признаки (v2.1):
- main_dish — основное блюда дня (категориальный)
- is_rainy, is_cold — погодные бинарные признаки
- temp_celsius, precipitation_mm — погодные числовые признаки
- present_avg_3d, present_avg_7d — скользящее среднее посещаемости
- present_trend — тренд посещаемости (сегодня минус вчера)
- present_t_minus_2, present_t_minus_3 — посещаемость 2 и 3 дня назад
- present_same_dow_last_week — посещаемость в этот же день недели на прошлой неделе
- is_holiday_calendar — праздник/каникулы из календаря

Категориальные признаки передаются в CatBoost напрямую (без One-Hot Encoding).
Вся конфигурация фичей — в features.py (единый источник правды).
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
from sqlalchemy import create_engine

warnings.filterwarnings("ignore", category=UserWarning, module="catboost")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

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
    MAIN_DISHES,
    DISH_POPULARITY,
    BUFFET_CATEGORIES,
    buffet_ratio_for,
    _buffet_category_id,
    get_holiday_calendar,
)

# Пути к моделям (версия v2)
MODELS_DIR = Path(__file__).parent / "models"
MODELS_DIR.mkdir(exist_ok=True)

MODEL_VERSION = "v2"

STANDARD_GLOBAL_MODEL_PATHS = {
    h: MODELS_DIR / f"standard_global_model_{MODEL_VERSION}_h{h}.cbm" for h in HORIZONS
}
BUFFET_MODEL_PATHS = {
    h: MODELS_DIR / f"buffet_model_{MODEL_VERSION}_h{h}.cbm" for h in HORIZONS
}

# URL подключения к PostgreSQL бэкенда
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+psycopg2://school:school@db:5432/school_balance",
)

# Популярные школы для дообучения
TOP_SCHOOLS_COUNT = 10


def set_seed(seed: int = 42) -> None:
    """Фиксация случайности для воспроизводимости."""
    np.random.seed(seed)


def load_backend_data(database_url: str | None = None) -> pd.DataFrame:
    """
    Загружает данные из БД бэкенда (PostgreSQL) и строит обучающую выборку.

    Для каждого дня с посещаемостью формируем запись со всеми признаками из features.py
    и таргетами по каждому горизонту h=1..5.
    """
    url = database_url or DATABASE_URL
    logger.info(f"Загрузка данных из PostgreSQL: {url}")

    try:
        engine = create_engine(url, pool_pre_ping=True)

        schools_df = pd.read_sql("SELECT * FROM schools", engine)
        logger.info(f"Школ: {len(schools_df)}")

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

        # Календарь каникул/праздников
        holiday_calendar = get_holiday_calendar()

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
            opt_outs = len(class_students[class_students["meal_status"] == "opt_out"])

            absent_ids = set(att.get("absent_student_ids", []) or [])
            regular_ids = set(class_students[class_students["meal_status"] == "regular"]["id"])
            absent = len(absent_ids & regular_ids)

            present_today = max(0, total_students - opt_outs - absent)

            # Present вчера, 2 и 3 дня назад
            present_yesterday = present_lookup.get((class_id, att_date - timedelta(days=1)), present_today)
            present_t_minus_2 = present_lookup.get((class_id, att_date - timedelta(days=2)), present_today)
            present_t_minus_3 = present_lookup.get((class_id, att_date - timedelta(days=3)), present_today)

            # Скользящее среднее за 3 и 7 дней (включая сегодня)
            window_3 = [present_lookup.get((class_id, att_date - timedelta(days=i)), present_today) for i in range(3)]
            window_7 = [present_lookup.get((class_id, att_date - timedelta(days=i)), present_today) for i in range(7)]
            present_avg_3d = float(np.mean(window_3))
            present_avg_7d = float(np.mean(window_7))
            present_trend = float(present_today - present_yesterday)

            # Посещаемость в этот же день недели на прошлой неделе
            present_same_dow_last_week = present_lookup.get(
                (class_id, att_date - timedelta(days=7)), present_today
            )

            # Календарные признаки
            month = att_date.month
            day_of_month = att_date.day
            day_of_week = att_date.weekday()  # 0=Пн, 6=Вс
            # День сам по себе праздничный/каникулярный
            is_holiday_calendar_flag = 1 if att_date in holiday_calendar else 0
            # День перед началом каникул/праздника (завтра уже каникулы)
            is_pre_vacation = 1 if (att_date + timedelta(days=1)) in holiday_calendar else 0
            # Предпраздничный день (пятница перед уик-эндом)
            is_pre_holiday = 1 if day_of_week == 4 else 0

            # Погодные признаки (в БД нет — нейтральные значения,
            # реальная погода будет подтягиваться бэкендом при инференсе)
            temp_celsius = 15.0
            precipitation_mm = 0.0
            is_rainy = 0
            is_cold = 0 if temp_celsius >= COLD_THRESHOLD else 1

            # Блюдо дня (в БД нет таблицы меню для fixed-школ — None)
            main_dish = None

            record = {
                "school_id": school_id,
                "class_id": class_id,
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
                "temp_celsius": temp_celsius,
                "precipitation_mm": precipitation_mm,
                "is_rainy": is_rainy,
                "is_cold": is_cold,
                "meal_plan": meal_plan,
                "date": att_date,
            }

            # Для canteen-школ добавляем продажи вчера по категориям
            if meal_plan == "canteen":
                y_date = att_date - timedelta(days=1)
                for cat in BUFFET_CATEGORIES:
                    cat_id = _buffet_category_id(cat)
                    col = f"sold_{cat}_yesterday"
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
        df[f"target_portions_h{h}"] = df[f"target_portions_h{h}"].fillna(df["present_today"])

    # Базовый таргет (h=1) для обратной совместимости
    df["target_portions_tomorrow"] = df["target_portions_h1"]

    # Буфетные таргеты по горизонтам — реальные продажи по категориям из canteen_sales.
    # Каждая категория предсказывается независимо (сумма не обязана быть равна 1).
    buffet_mask = df["meal_plan"] == "canteen"
    if buffet_mask.any() and not sales_df.empty:
        sales_lookup = {
            (row["school_id"], row["date"], row["category_id"]): int(row["qty"])
            for _, row in sales_df.iterrows()
        }
        n_buffet = int(buffet_mask.sum())
        for h in HORIZONS:
            target_dates = df.loc[buffet_mask, "date"] + timedelta(days=h)
            schools_h = df.loc[buffet_mask, "school_id"]
            for cat in BUFFET_CATEGORIES:
                cat_id = _buffet_category_id(cat)
                col = f"target_{cat}_h{h}"
                keys = list(zip(schools_h, target_dates, [cat_id] * n_buffet))
                df.loc[buffet_mask, col] = [sales_lookup.get(k, 0) for k in keys]
    elif buffet_mask.any():
        # Нет таблицы продаж — нулевые таргеты по категориям
        for h in HORIZONS:
            for cat in BUFFET_CATEGORIES:
                df.loc[buffet_mask, f"target_{cat}_h{h}"] = 0

    logger.info(f"Построено записей для обучения: {len(df)}")
    return df


def _generate_synthetic_fallback() -> pd.DataFrame:
    """Генерация синтетических данных с новыми признаками (погода, календарь, лаги)."""
    logger.info("Генерация синтетических данных (fallback, v2.1)...")
    np.random.seed(42)

    # Школы не захардкожены — генерируются параметризованно из env.
    # SYNTHETIC_SCHOOLS_COUNT — число школ, SYNTHETIC_CLASSES_PER_SCHOOL — классов на школу.
    synth_schools_count = int(os.getenv("SYNTHETIC_SCHOOLS_COUNT", "6"))
    synth_classes_per_school = int(os.getenv("SYNTHETIC_CLASSES_PER_SCHOOL", "6"))

    schools = []
    classes_per_school = {}
    for i in range(1, synth_schools_count + 1):
        sid = f"school-{i}"
        meal_plan = "canteen" if i % 2 == 0 else "fixed"
        baseline = round(0.85 + 0.05 * (i % 4), 2)
        schools.append({
            "school_id": sid,
            "name": f"Школа №{i}",
            "meal_plan": meal_plan,
            "baseline": baseline,
            "city": "",
            "address": "",
        })
        classes_per_school[sid] = [
            (f"{g}{l}", ag, size)
            for g, l, ag, size in [
                ("1", "А", "junior", 25), ("2", "Б", "junior", 24),
                ("5", "А", "middle", 28), ("6", "Б", "middle", 27),
                ("9", "А", "senior", 26), ("10", "Б", "senior", 25),
            ][:synth_classes_per_school]
        ]

    start_date = date(2026, 9, 1)
    end_date = date(2026, 12, 18)
    all_dates = []
    d = start_date
    while d <= end_date:
        if d.weekday() < 5:
            all_dates.append(d)
        d += timedelta(days=1)

    holiday_calendar = get_holiday_calendar()

    month_factor = {9: 1.05, 10: 1.02, 11: 1.00, 12: 0.92}
    dow_factor = {0: 1.02, 1: 1.00, 2: 0.98, 3: 0.95, 4: 0.85}
    age_factor_target = {"junior": 0.90, "middle": 1.00, "senior": 1.10}

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

            present_history: list[int] = []
            for att_date in all_dates:
                dow = att_date.weekday()
                month = att_date.month

                is_holiday_calendar_flag = 1 if att_date in holiday_calendar else 0
                is_pre_vacation = 1 if (att_date + timedelta(days=1)) in holiday_calendar else 0
                is_pre_holiday = 1 if dow == 4 else 0

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
                holiday_factor = 0.70 if is_holiday_calendar_flag else 1.0

                base_present = (
                    (total_students - opt_outs)
                    * baseline
                    * month_factor.get(month, 1.0)
                    * dow_factor.get(dow, 1.0)
                    * age_factor_target[age_group]
                    * weather_factor
                    * holiday_factor
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
                    "is_holiday_calendar": is_holiday_calendar_flag,
                    "main_dish": main_dish,
                    "total_students": total_students,
                    "opt_outs": opt_outs,
                    "present_today": present_today,
                    "present_yesterday": present_history[-2] if len(present_history) >= 2 else present_today,
                    "present_t_minus_2": present_history[-3] if len(present_history) >= 3 else present_today,
                    "present_t_minus_3": present_history[-4] if len(present_history) >= 4 else present_today,
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

    # present_same_dow_last_week
    df["present_same_dow_last_week"] = df.groupby(["school_id", "class_id"])["present_today"].shift(7)

    # present_yesterday, present_t_minus_2, present_t_minus_3 через shift
    df["present_yesterday"] = df.groupby(["school_id", "class_id"])["present_today"].shift(1)
    df["present_t_minus_2"] = df.groupby(["school_id", "class_id"])["present_today"].shift(2)
    df["present_t_minus_3"] = df.groupby(["school_id", "class_id"])["present_today"].shift(3)

    # Заполняем NaN для первых дней
    for col in ("present_yesterday", "present_t_minus_2", "present_t_minus_3", "present_same_dow_last_week"):
        first_mask = df[col].isna()
        df.loc[first_mask, col] = df.loc[first_mask, "present_today"]

    # Таргеты по горизонтам
    for h in HORIZONS:
        df[f"target_portions_h{h}"] = df.groupby(["school_id", "class_id"])["present_today"].shift(-h)
        df[f"target_portions_h{h}"] = df[f"target_portions_h{h}"].fillna(df["present_today"])
    df["target_portions_tomorrow"] = df["target_portions_h1"]

    # Буфетные фичи и таргеты для canteen-школ.
    # Каждая категория независима — сумма не обязана быть равна 1.
    buffet_mask = df["meal_plan"] == "canteen"
    if buffet_mask.any():
        n_buffet = int(buffet_mask.sum())
        for h in HORIZONS:
            target_col = f"target_portions_h{h}"
            for cat in BUFFET_CATEGORIES:
                # Таргет категории = доля категории * общий таргет (симмулирует спрос)
                ratios = df.loc[buffet_mask, "age_group"].map(
                    lambda ag: buffet_ratio_for(cat, ag)
                )
                noise = np.random.uniform(0.85, 1.15, size=n_buffet)
                df.loc[buffet_mask, f"target_{cat}_h{h}"] = (
                    df.loc[buffet_mask, target_col] * ratios * noise
                ).clip(lower=0).astype(int)

        # Продажи вчера для canteen по категориям
        for cat in BUFFET_CATEGORIES:
            col = f"sold_{cat}_yesterday"
            ratios = df.loc[buffet_mask, "age_group"].map(
                lambda ag: buffet_ratio_for(cat, ag)
            )
            noise = np.random.uniform(0.85, 1.15, size=n_buffet)
            df.loc[buffet_mask, col] = (
                df.loc[buffet_mask, "present_yesterday"] * ratios * noise
            ).clip(lower=0).astype(int)

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
    X_train, y_train, X_test, y_test, horizon: int, model_type: str = "standard"
) -> CatBoostRegressor:
    """Обучает глобальную модель CatBoostRegressor для заданного горизонта."""
    loss = "RMSE" if model_type == "standard" else "MultiRMSE"
    eval_metric = "RMSE" if model_type == "standard" else "MultiRMSE"
    logger.info(f"Обучение глобальной {model_type} модели (h={horizon}, {loss})...")

    train_pool = _make_pool(X_train, CAT_FEATURES, label=y_train)
    test_pool = _make_pool(X_test, CAT_FEATURES, label=y_test)

    model = CatBoostRegressor(
        iterations=600,
        learning_rate=0.05,
        depth=6,
        loss_function=loss,
        eval_metric=eval_metric,
        random_seed=42,
        verbose=100,
        early_stopping_rounds=50,
        allow_writing_files=False,
    )
    model.fit(train_pool, eval_set=test_pool, use_best_model=True)

    logger.info(
        f"Глобальная {model_type} модель h={horizon}: best_iteration={model.best_iteration_}, "
        f"best_score={model.best_score_['validation'][eval_metric]:.4f}"
    )
    return model


def _fit_school_model(
    school_id: str,
    X_school: pd.DataFrame,
    y_school: pd.Series | pd.DataFrame,
    dates_school: pd.Series,
    global_model: CatBoostRegressor,
    horizon: int,
    model_type: str = "standard",
) -> CatBoostRegressor | None:
    """Дообучение модели для конкретной школы через Transfer Learning."""
    if len(X_school) < 10:
        logger.warning(f"Для {school_id} h={horizon} мало данных ({len(X_school)}), пропускаем")
        return None

    loss = "RMSE" if model_type == "standard" else "MultiRMSE"

    # Временной сплит: ранние даты на обучение, поздние — на валидацию (без утечки)
    order = dates_school.argsort(kind="mergesort")
    n = len(order)
    n_train = int(n * 0.8)
    if n_train < 1 or n - n_train < 1:
        # Недостаточно данных для валидации — обучаем на всём
        X_s_train, X_s_test = X_school, X_school.iloc[:0]
        y_s_train, y_s_test = y_school, y_school.iloc[:0]
    else:
        train_idx = order[:n_train]
        test_idx = order[n_train:]
        X_s_train = X_school.iloc[train_idx]
        X_s_test = X_school.iloc[test_idx]
        y_s_train = y_school.iloc[train_idx]
        y_s_test = y_school.iloc[test_idx]

    train_pool = _make_pool(X_s_train, CAT_FEATURES, label=y_s_train)
    test_pool = _make_pool(X_s_test, CAT_FEATURES, label=y_s_test)

    fine_tuned = CatBoostRegressor(
        iterations=300,
        learning_rate=0.03,
        depth=6,
        loss_function=loss,
        eval_metric=loss,
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
        f"{school_id} {model_type} h={horizon}: best_iteration={fine_tuned.best_iteration_}, "
        f"best_score={fine_tuned.best_score_['validation'][loss]:.4f}"
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

        # Временной сплит: обучение на ранних датах, валидация на поздних (без утечки из будущего)
        order = df["date"].argsort(kind="mergesort")
        n = len(order)
        n_train = int(n * 0.8)
        train_idx = order[:n_train]
        test_idx = order[n_train:]
        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        # 1. Глобальная модель для этого горизонта
        global_model = _fit_global_model(X_train, y_train, X_test, y_test, h, model_type="standard")
        global_models[h] = global_model

        # Сохранение глобальной модели
        global_model.save_model(str(STANDARD_GLOBAL_MODEL_PATHS[h]))
        logger.info(f"Глобальная модель h={h}: {STANDARD_GLOBAL_MODEL_PATHS[h]}")

        # 2. Дообучение для ВСЕХ школ с достаточным количеством данных
        all_school_ids = X["school_id"].unique().tolist()
        logger.info(f"Школы для дообучения h={h}: {all_school_ids}")

        school_models_h: dict[int, CatBoostRegressor] = {}
        for school_id in all_school_ids:
            mask = X["school_id"] == school_id
            model = _fit_school_model(
                school_id, X[mask], y[mask], df.loc[mask, "date"], global_model, h, model_type="standard"
            )
            if model is not None:
                school_models_h[school_id] = model

        # Сохранение школьных моделей
        for school_id, model in school_models_h.items():
            model_path = MODELS_DIR / f"{school_id}_model_{MODEL_VERSION}_h{h}.cbm"
            model.save_model(str(model_path))
            logger.info(f"Модель для {school_id} h={h}: {model_path}")

        # Аккумулируем школьные модели
        for school_id, model in school_models_h.items():
            school_models_all.setdefault(school_id, {})[h] = model

    return global_models, school_models_all


def train_buffet_models(df: pd.DataFrame) -> tuple:
    """
    Обучение буфетных моделей по всем горизонтам h=1..5.
    Возвращает (dict horizon->global_model, dict school_id->dict horizon->model).
    """
    logger.info("=" * 60)
    logger.info("ОБУЧЕНИЕ МОДЕЛИ 'ШВЕДСКИЙ СТОЛ' (многогоризонт, h=1..5)")
    logger.info("=" * 60)

    global_buffet_models: dict[int, CatBoostRegressor] = {}
    school_buffet_models_all: dict[str, dict[int, CatBoostRegressor]] = {}

    buffet_df = df[df["meal_plan"] == "canteen"].copy()
    if buffet_df.empty:
        logger.warning("Нет canteen-данных — буфетные модели не обучены")
        return global_buffet_models, school_buffet_models_all

    for h in HORIZONS:
        target_cols = [f"target_{cat}_h{h}" for cat in BUFFET_CATEGORIES]
        logger.info(f"\n--- Горизонт h={h} (таргеты: {target_cols}) ---")

        X = buffet_df[BUFFET_INPUT_FEATURES]
        y = buffet_df[target_cols]

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

        logger.info(
            f"Буфетная глобальная модель h={h}: best_iteration={model.best_iteration_}"
        )
        model.save_model(str(BUFFET_MODEL_PATHS[h]))
        logger.info(f"Буфетная глобальная модель h={h}: {BUFFET_MODEL_PATHS[h]}")
        global_buffet_models[h] = model

        # 2. Дообучение для ВСЕХ canteen-школ с достаточным количеством данных
        all_school_ids = X["school_id"].unique().tolist()
        logger.info(f"Canteen-школы для дообучения h={h}: {all_school_ids}")

        school_models_h: dict[int, CatBoostRegressor] = {}
        for school_id in all_school_ids:
            mask = X["school_id"] == school_id
            model = _fit_school_model(
                school_id,
                X[mask],
                y[mask],
                buffet_df.loc[mask, "date"],
                model,
                h,
                model_type="buffet",
            )
            if model is not None:
                school_models_h[school_id] = model

        # Сохранение буфетных школьных моделей
        for school_id, model in school_models_h.items():
            model_path = MODELS_DIR / f"{school_id}_buffet_model_{MODEL_VERSION}_h{h}.cbm"
            model.save_model(str(model_path))
            logger.info(f"Буфетная модель для {school_id} h={h}: {model_path}")

        # Аккумулируем школьные модели
        for school_id, model in school_models_h.items():
            school_buffet_models_all.setdefault(school_id, {})[h] = model

    return global_buffet_models, school_buffet_models_all


def main() -> None:
    logger.info("Запуск пайплайна обучения моделей 'Школьный Баланс-Пит' v2.1")
    set_seed(42)

    df = load_backend_data()

    # 1. Стандартное питание (многогоризонт)
    global_models, school_models_all = train_standard_models(df)

    # 2. Шведский стол (многогоризонт, с дообучением по школам)
    global_buffet_models, school_buffet_models_all = train_buffet_models(df)

    logger.info("=" * 60)
    logger.info("ОБУЧЕНИЕ ЗАВЕРШЕНО УСПЕШНО")
    logger.info(f"Версия модели: {MODEL_VERSION}")
    logger.info(f"Горизонты: {HORIZONS}")
    logger.info(f"Модели: {MODELS_DIR}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
