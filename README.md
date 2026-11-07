# AI-сервис школьного питания (Школьный Баланс-Пит)

Микросервис для предсказания количества порций школьного питания на основе ML-моделей CatBoost.
Интегрируется с бэкендом (`/backend/`) через ML Service API.

## Что исправлено относительно документации

Документация ранее не соответствовала реализации. Основные исправления:

* Категории шведского стола стали динамическими (регулируются переменной окружения `BUFFET_CATEGORIES`), сумма долей категорий не обязана быть равна 1.0 — каждый ученик берёт то, что хочет (кто-то ничего, кто-то две котлеты).
* Возрастная группа (`age_group`) теперь передаётся в интеграционный эндпоинт `/predict` и обязательно учитывается моделью.
* Календарные признаки разделены: `is_holiday_calendar` (сам день в каникулах/праздниках), `is_pre_vacation` (день перед каникулами), `is_pre_holiday` (пятница).
* Таргеты буфера при обучении — реальные продажи по категориям из `canteen_sales`, а не выдуманные фиксированные доли.
* Устранена утечка данных при обучении: сплит по времени (ранние даты на обучение, поздние — на валидацию), без случайного перемешивания временных рядов.
* Сервис не накладывает автоматический буфер на итоговый прогноз — буфер добавляет фронтенд при генерации заявки.
* Удалены старые модели без суффикса `_v2`; загрузчик берёт только `*_model_v2_h*.cbm`.

## Структура проекта

```
ai-service/
├── train.py                    # Скрипт обучения моделей (данные из БД бэкенда)
├── main.py                     # FastAPI-сервис
├── features.py                 # Единый источник правды для признаков
├── requirements.txt            # Зависимости Python
├── Dockerfile                  # Docker образ
├── docker-compose.yml          # Docker Compose
├── smoke_test.py               # Проверочный скрипт
├── models/                     # Обученные модели (только *_v2_h*.cbm)
│   ├── standard_global_model_v2_h{1..5}.cbm
│   ├── school-{id}_model_v2_h{1..5}.cbm
│   └── school-{id}_buffet_model_v2_h{1..5}.cbm
└── README.md                   # Эта документация
```

Модельные файлы `.cbm` не отслеживаются гитом (в `.gitignore`), обучаются через `train.py`.

## Школы

Школы не захардкожены — список берётся из базы бэкенда (таблица `schools`). Любая школа, зарегистрированная в бэкенде, автоматически доступна для прогнозирования. При отсутствии дообученной модели для школы используется глобальная модель как fallback.

## Переменные окружения

| Переменная | Описание | По умолчанию |
|------------|----------|--------------|
| `BUFFET_CATEGORIES` | Категории шведского стола через запятую | `hot,salad,bakery` |
| `HOLIDAY_CALENDAR` | Дополнительные даты каникул (JSON-массив ISO) | встроенный набор |
| `DATABASE_URL` | URL PostgreSQL бэкенда | `postgresql+psycopg2://school:school@db:5432/school_balance` |
| `ML_SERVICE_URL` | URL для бэкенда | `http://ai-service:8000` |

Пример расширения категорий буфера:
```
BUFFET_CATEGORIES=hot,salad,bakery,soup,dessert
```

## Шаги запуска

### 1. Обучите модели

```bash
cd ai-service
python3 train.py
```

Данные загружаются из PostgreSQL бэкенда (таблицы `schools`, `classes`, `students`, `attendance`, `canteen_sales`).

Обучаются:
* `standard_global_model_v2_h{h}.cbm` — глобальные модели по горизонтам h=1..5
* `school-{id}_model_v2_h{h}.cbm` — дообученные модели школ через Transfer Learning (`init_model=global_model`)
* `school-{id}_buffet_model_v2_h{h}.cbm` — дообученные буфетные модели школ
* `buffet_model_v2_h{h}.cbm` — глобальная мультитаргетная модель шведского стола

Обучение использует временной сплит (ранние даты — на обучение, поздние — на валидацию), без утечки будущего в прошлое.

### 2. Запустите сервер

**Локально:**
```bash
python3 main.py
```

**Через Docker Compose:**
```bash
docker compose build
docker compose up -d
```

Сервер будет доступен на `http://localhost:8000` (через compose пробрасывается на `8001:8000`).

## API Эндпоинты

### `POST /predict/standard` — стандартное питание (прямой вызов)

Прогноз количества порций на 1 день вперёд. Возрастная группа учитывается.

```json
{
  "school_id": "school-a",
  "age_group": "middle",
  "month": 10, "day_of_month": 15, "day_of_week": 2,
  "is_pre_holiday": 0, "is_pre_vacation": 0, "is_holiday_calendar": 0,
  "main_dish": "гречка с котлетой",
  "total_students": 300, "opt_outs": 24,
  "present_today": 276, "present_yesterday": 270,
  "present_t_minus_2": 268, "present_t_minus_3": 265,
  "present_avg_3d": 272.0, "present_avg_7d": 268.0,
  "present_trend": 6.0, "present_same_dow_last_week": 274,
  "temp_celsius": 12.0, "precipitation_mm": 0.0,
  "is_rainy": 0, "is_cold": 0
}
```

Ответ:
```json
{ "predicted_portions": 25.74, "confidence": 0.73 }
```

### `POST /predict/multi` — стандартное питание на несколько дней (1–5)

```json
{
  "school_id": "school-a",
  "age_group": "middle",
  "total_students": 300, "opt_outs": 24,
  "present_today": 276, "present_yesterday": 270,
  "present_t_minus_2": 268, "present_t_minus_3": 265,
  "present_avg_3d": 272.0, "present_avg_7d": 268.0,
  "present_trend": 6.0, "present_same_dow_last_week": 274,
  "days": [
    { "date": "2026-10-16", "month": 10, "day_of_month": 16, "day_of_week": 3,
      "is_pre_holiday": 0, "is_pre_vacation": 0, "is_holiday_calendar": 0,
      "main_dish": null, "total_students": 300, "opt_outs": 24,
      "present_today": 276, "present_yesterday": 270,
      "present_t_minus_2": 268, "present_t_minus_3": 265,
      "present_avg_3d": 272.0, "present_avg_7d": 268.0,
      "present_trend": 6.0, "present_same_dow_last_week": 274,
      "temp_celsius": 11.0, "precipitation_mm": 0.0, "is_rainy": 0, "is_cold": 0 }
  ]
}
```

Ответ:
```json
{ "school_id": "school-a", "predictions": [
  { "date": "2026-10-16", "predicted_portions": 25.7, "confidence": 0.73 }
]}
```

### `POST /predict/buffet` — шведский стол (прямой вызов)

Прогноз по категориям блюд. Категории настраиваются через `BUFFET_CATEGORIES`. Сумма категорий не обязана быть равна 1.0.

```json
{
  "school_id": "school-b",
  "age_group": "middle",
  "month": 10, "day_of_month": 15, "day_of_week": 2,
  "is_pre_holiday": 0, "is_pre_vacation": 0, "is_holiday_calendar": 0,
  "main_dish": null,
  "total_students": 200, "opt_outs": 16,
  "present_today": 184, "present_yesterday": 180,
  "present_t_minus_2": 176, "present_t_minus_3": 172,
  "present_avg_3d": 182.0, "present_avg_7d": 178.0,
  "present_trend": 4.0, "present_same_dow_last_week": 182,
  "temp_celsius": 12.0, "precipitation_mm": 0.0,
  "is_rainy": 0, "is_cold": 0,
  "sold_hot_yesterday": 70, "sold_salad_yesterday": 30, "sold_bakery_yesterday": 45
}
```

Ответ (дефолтные категории):
```json
{ "categories": { "hot": 10.82, "salad": 4.33, "bakery": 5.34 } }
```

### `POST /predict` — интеграция с бэкендом (ML Service API)

Бэкенд (`backend/app/services/ml_client.py`) отправляет на этот эндпоинт полный расчёт прогноза (`ForecastOut`) и ожидает скорректированные поля `total_portions` и `confidence`, а также прогноз на несколько дней.

**Поля запроса:**
* `school_id` (обяз.), `school_name`, `age_group` (junior/middle/senior, по умолчанию `middle`)
* `date`, `meal_plan` (`fixed` / `canteen`), `buffer_percent` (информационное, буфер не накладывается сервисом)
* `total_students`, `opt_outs`, `absent`, `present` (обяз.), `total_portions`
* `by_age_group`, `by_category`, `marked_classes`, `total_classes`, `based_on_date`, `confidence`
* `forecast_days` (1–5), `main_dish`, `weather` (погода по ISO-датам)
* `present_avg_3d`, `present_avg_7d`, `present_trend`, `present_yesterday`, `present_t_minus_2`, `present_t_minus_3`, `present_same_dow_last_week`, `is_holiday_calendar`

**Пример запроса (fixed meal plan):**
```json
{
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
  "main_dish": null,
  "weather": {
    "2026-10-16": { "temp_celsius": 11.0, "precipitation_mm": 0.0, "is_rainy": 0, "is_cold": 0 },
    "2026-10-17": { "temp_celsius": 10.0, "precipitation_mm": 2.0, "is_rainy": 1, "is_cold": 0 }
  },
  "present_avg_3d": 272.0,
  "present_avg_7d": 268.0,
  "present_trend": 6.0,
  "present_yesterday": 270,
  "present_t_minus_2": 268,
  "present_t_minus_3": 265,
  "present_same_dow_last_week": 274,
  "is_holiday_calendar": 0
}
```

**Пример запроса (canteen meal plan):**
```json
{
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
  "main_dish": null,
  "weather": {
    "2026-10-16": { "temp_celsius": 11.0, "precipitation_mm": 0.0, "is_rainy": 0, "is_cold": 0 }
  },
  "present_avg_3d": 182.0,
  "present_avg_7d": 178.0,
  "present_trend": 4.0,
  "present_yesterday": 180,
  "present_t_minus_2": 176,
  "present_t_minus_3": 172,
  "present_same_dow_last_week": 182,
  "is_holiday_calendar": 0
}
```

Ответ:
```json
{
  "total_portions": 26,
  "confidence": 0.73,
  "forecast_days": [
    { "date": "2026-10-17", "horizon": 1, "predicted_portions": 26, "confidence": 0.73 },
    { "date": "2026-10-18", "horizon": 2, "predicted_portions": 27, "confidence": 0.71 }
  ],
  "model_used": "ml",
  "categories": { "hot": 10.9, "salad": 4.4, "bakery": 5.4 }
}
```

Поле `categories` заполнено только для `meal_plan=canteen`. `buffer_percent` не применяется сервисом — фронтенд добавляет буфер при генерации заявки.

## Календарные признаки

* `is_holiday_calendar` — 1, если дата сама по себе в календаре каникул/праздников.
* `is_pre_vacation` — 1, если следующий день (`date + 1`) в календаре каникул.
* `is_pre_holiday` — 1, если день неделя пятница (`day_of_week == 4`).

## Архитектура обучения (Transfer Learning)

1. **Глобальная модель** обучается на данных всех школ из БД бэкенда.
2. Для каждой школы с достаточным количеством данных выполняется **дообучение** через `init_model=global_model` (CatBoost Transfer Learning).
3. Дообученные модели сохраняются как `school-{id}_model_v2_h{h}.cbm` (стандарт) и `school-{id}_buffet_model_v2_h{h}.cbm` (буфет).
4. При предсказании сервис выбирает модель школы по `school_id`, с **fallback на глобальную модель** для неизвестных школ.
5. Модель «Шведский стол» — одна мультитаргетная модель с `loss_function='MultiRMSE'` по категориям из `BUFFET_CATEGORIES`.
6. Таргеты буфера — реальные продажи по категориям из `canteen_sales` за `date + h`, а не фиксированные доли.
7. Сплит при обучении — временной (без перемешивания), чтобы исключить утечку будущих данных в обучающую выборку.

## Инкрементальное дообучение

При повторном запуске `train.py` обучение не начинается с нуля, а достраивается на уже имеющихся моделях:

1. **Отслеживание последнего обучения.** После успешного обучения в директории `models/` создаётся файл `.last_train`, в котором записана ISO-дата последнего успешного запуска обучения (например, `2026-11-07`).

2. **Загрузка существующих моделей как `init_model`.** Существующие `.cbm` модели (`standard_global_model_v2_h{h}`, `school-{id}_model_v2_h{h}`, `buffet_model_v2_h{h}`, `school-{id}_buffet_model_v2_h{h}`) загружаются через `CatBoostRegressor().load_model()` и передаются в обучение как `init_model`, чтобы новые модели продолжали обучение с того же состояния.

3. **Фильтрация данных по дате.** Обучение ведётся только на новых данных, поступивших после последней метки даты: `df[df["date"] > last_train_date]`. Если новых данных нет — обучение пропускается, чтобы не тратить ресурсы без необходимости.

4. **Мягкие гиперпараметры для дообучения.** Для инкрементального дообучения используются более бережные гиперпараметры CatBoost: `iterations=150`, `learning_rate=0.02`, `depth=6`, `early_stopping_rounds=30`, `use_best_model=True` (вместо `iterations=600`, `learning_rate=0.05` при первичном обучении).

5. **Первичное обучение с нуля.** При полном отсутствии моделей в `models/` обучение выполняется с нуля на всех данных, как при первом запуске.

**Рекомендация.** Запускайте `train.py` по расписанию (cron или CronJob в Kubernetes) для постепенной адаптации моделей к новым данным без полного переобучения.

## Категориальные признаки

`school_id`, `age_group`, `month`, `day_of_month`, `day_of_week`, `main_dish` передаются в CatBoost напрямую через `Pool(cat_features=...)`, без ручного One-Hot Encoding.

## Интеграция с бэкендом

В `.env` бэкенда укажите:
```
ML_SERVICE_URL=http://localhost:8000
```

Затем бэкенд будет автоматически обращаться к `POST /predict` для ML-коррекции прогнозов.

URL для бэкенда через внутреннюю сеть docker compose: `http://ai-service:8000/predict`.

## Запуск в Docker

Через корневой docker compose (рекомендуется):

```bash
cd /Users/kretoffer/lyceum/lbsuhack-2060
docker compose up --build
```

AI-сервис доступен на http://localhost:8001 (внутри контейнера порт 8000).

Отдельно (локальный compose):

```bash
cd ai-service
docker compose up -d
# доступен на http://localhost:8001
```

## Проверка

```bash
cd ai-service
python3 -m pytest tests/      # модульные тесты
python3 smoke_test.py         # дымовые тесты эндпоинтов
```

---

*Сервис разработан в рамках проекта «Школьный Баланс-Пит»*
