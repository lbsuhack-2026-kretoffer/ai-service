# AI-сервис школьного питания (Школьный Баланс-Пит)

Микросервис для предсказания количества порций школьного питания на основе ML-моделей CatBoost.
Интегрируется с бэкендом (`/backend/`) через ML Service API.

## Структура проекта

```
ai-service/
├── train.py                    # Скрипт обучения моделей (данные из БД бэкенда)
├── main.py                     # FastAPI-бэкенд сервис
├── requirements.txt            # Зависимости Python
├── Dockerfile                  # Docker образ
├── docker-compose.yml          # Docker Compose
├── smoke_test.py               # Проверочный скрипт
├── models/                     # Обученные модели
│   ├── standard_global_model.cbm          # Глобальная модель (все школы)
│   ├── school-a_model.cbm                  # Дообученная модель школы-A (Transfer Learning)
│   ├── school-b_model.cbm                  # Дообученная модель школы-B (Transfer Learning)
│   └── buffet_model.cbm                    # Шведский стол (MultiRMSE)
└── README.md                   # Эта документация
```

## Школы в бэкенде

| ID школы | Название | meal_plan | Модель |
|----------|----------|-----------|--------|
| `school-a` | Гимназия №12 | `fixed` (стандартное) | `school-a_model.cbm` |
| `school-b` | Средняя школа №34 | `canteen` (шведский стол) | `school-b_model.cbm` |

Неизвестные школы используют **глобальную модель** (`standard_global_model.cbm`) как fallback.

## Шаги запуска

### 1. Обучите модели

```bash
cd ai-service
python3 train.py
```

*Данные загружаются из `../backend/school_balance.db`. Обучаются:*
- `standard_global_model.cbm` — глобальная модель на всех школах
- `school-a_model.cbm`, `school-b_model.cbm` — дообученные модели через Transfer Learning (`init_model`)
- `buffet_model.cbm` — мультитаргетная модель шведского стола

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

Сервер будет доступен на `http://localhost:8000`.

## API Эндпоинты

### `POST /predict/standard` — стандартное питание (прямой вызов)

```json
{
  "school_id": "school-a",
  "age_group": "middle",
  "month": 10, "day_of_month": 15, "day_of_week": 2,
  "is_pre_holiday": 0, "is_pre_vacation": 0,
  "total_students": 300, "opt_outs": 24,
  "present_today": 276, "present_yesterday": 270
}
```
Ответ: `23.4` (число порций)

### `POST /predict/buffet` — шведский стол (прямой вызов)

```json
{
  "school_id": "school-b",
  "age_group": "middle",
  "month": 10, "day_of_month": 15, "day_of_week": 2,
  "is_pre_holiday": 0, "is_pre_vacation": 0,
  "total_students": 200, "opt_outs": 16,
  "present_today": 184, "present_yesterday": 180,
  "sold_hot_yesterday": 70, "sold_salad_yesterday": 30, "sold_bakery_yesterday": 45
}
```
Ответ:
```json
{ "hot": 9.8, "salad": 3.9, "bakery": 4.9 }
```

### `POST /predict` — интеграция с бэкендом (ML Service API)

Бэкенд (`backend/app/services/ml_client.py`) отправляет на этот эндпоинт полный расчёт прогноза (`ForecastOut`) и ожидает скорректированные поля `total_portions` и `confidence`.

**Пример запроса (fixed meal plan):**
```json
{
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
  "confidence": 0.8
}
```

**Пример запроса (canteen meal plan):**
```json
{
  "school_id": "school-b",
  "meal_plan": "canteen",
  "buffer_percent": 8,
  "total_students": 200,
  "opt_outs": 16,
  "present": 184,
  "total_portions": 199,
  "yesterday_total_qty": 88,
  ...
}
```

Ответ:
```json
{ "total_portions": 23, "confidence": 0.8 }
```

## Архитектура обучения (Transfer Learning)

1. **Глобальная модель** обучается на данных всех школ из БД бэкенда.
2. Для каждой школы с достаточным количеством данных выполняется **дообучение** через `init_model=global_model` (CatBoost Transfer Learning).
3. Дообученные модели сохраняются как `school-{id}_model.cbm`.
4. При предсказании сервис выбирает модель школы по `school_id`, с **fallback на глобальную модель** для неизвестных школ.
5. Модель «Шведский стол» — одна мультитаргетная модель с `loss_function='MultiRMSE'`.

## Категориальные признаки

`school_id`, `age_group`, `month`, `day_of_month`, `day_of_week` передаются в CatBoost напрямую через `Pool(cat_features=...)`, без ручного One-Hot Encoding.

## Интеграция с бэкендом

В `.env` бэкенда укажите:
```
ML_SERVICE_URL=http://localhost:8000
```

Затем бэкенд будет автоматически обращаться к `POST /predict` для ML-коррекции прогнозов.

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
# достунен на http://localhost:8001
```

## Интеграция с бэкендом

Бэкенд (`backend/app/services/ml_client.py`) отправляет на этот эндпоинт полный расчёт прогноза (`ForecastOut`) и ожидает скорректированные поля `total_portions` и `confidence` через `POST /predict`.

URL для бэкенда: `http://ai-service:8000/predict` (через внутреннюю сеть docker compose).

---

*Сервис разработан в рамках проекта «Школьный Баланс-Пит»*