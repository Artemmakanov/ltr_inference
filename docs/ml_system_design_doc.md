# ML System Design Doc: Real-time Movie Re-ranker

## 1. Goal & Overview

**Business Goal:** Улучшить персонализацию выдачи фильмов для пользователей
**Technical Goal:** Создать Low-Latency сервис ранжирования (Learning-to-Rank), оптимизированный для работы в ограниченных ресурсах (CPU Inference), и продемонстрировать полный цикл MLOps: от подготовки данных до инференса.

## 2. Metrics & Constraints

### 2.1 Offline Metrics (Quality)

* **HIT@5 (Hit rate at 5):** Основная метрика ранжирования. Показывает, насколько "правильные" фильмы находятся в топе выдачи.

### 2.2 System Metrics (Performance)

* **Latency (P95):** < 50ms (на батч из 100 кандидатов). Критически важно для Real-time системы.
* **Model Size:** < 100 MB (для быстрой загрузки и деплоя).
* **Throughput:** > 100 RPS на одном инстансе (CPU).

### 2.3 Constraints

* **Hardware:** Training – GPU; Inference – CPU (Mac/Docker).
* **Data:** MovieLens 1M (Static snapshot).

## 3. Architecture

Система реализует классическую **Two-Stage Architecture**:

1. **Candidate Generation (Retrieval):**
* *Input:* User ID.
* *Logic:* Быстрый отбор ~100 релевантных кандидатов из тысяч.
* *Algorithm:* Faiss (ANN search) по эмбеддингам (ALS).


2. **Re-ranking (Scoring):**
* *Input:* User Features + List of Candidates.
* *Logic:* Пересчет скора для каждого кандидата с учетом сложных фичей.
* *Model:* **CatBoostRanker (YetiRank)**

## 4. Data Pipeline & Features

### 4.1 Dataset

* **Source:** MovieLens 1M.
* **Split Strategy:** Global Time Split (Train: 80%, Test: 20% by Timestamp).

### 4.2 Feature Engineering (Feature Store)

Фичи будут рассчитываться оффлайн (Polars) и загружаться в Redis для инференса.

| Feature Group | Features | Type | Storage |
| --- | --- | --- | --- |
| **User** | Gender, Age, Occupation, Zip-code | Categorical | Redis (Hash) |
| **Item** | Genres, Year (extracted from Title) | Text/Cat | Redis (Hash) |
| **Interaction** | User-Item Rating (Target) | Int (0/1) | None (Train only) |

* **Target Definition:** `Rel(u, i) = 1` if Rating > 3, else `0`.

## 5. Modeling Strategy

* **Algorithm:** Gradient Boosting on Decision Trees (CatBoost).
* **Objective:** `YetiRank` (Listwise loss optimization). Это SOTA подход, который оптимизирует напрямую метрики ранжирования, а не ошибку классификации.
* **Parameters:**
* `group_id`: UserID.
* `depth`: 6-8.
* `task_type`: CPU (Training).


## 6. Inference & Optimization

Это ключевая часть проекта. Задача — минимизировать Latency.

1. **Framework:** FastAPI (Python) + Gunicorn/Uvicorn.
2. **Runtime Optimization:**
* Экспорт CatBoost модели в формат **ONNX**.
* Использование `onnxruntime` вместо нативного `.predict()`.
* *Experiment:* Сравнение Latency (Native vs ONNX).


3. **Data Access:**
* Использование Redis Pipeline (`MGET`) для одновременного получения фичей всех 100 кандидатов.
* Векторизация подготовки батча через Polars/Pandas (без циклов Python).


## 7. Результаты обучения и артефакты

В этом разделе фиксируются базовые показатели модели, обученной на MovieLens 1M.

### 7.1 Offline Quality (Метрики качества)

* **HIT@5:** `` (Доля случаев, когда релевантный фильм попал в топ-5 рекомендаций).
* **Время обучения (GPU):** `` (CatBoost YetiRank, 1M транзакций, глубина 6, iteration=5000).

### 7.2 Характеристики артефактов

| Артефакт | Формат | Размер | Описание |
| --- | --- | --- | --- |
| **Ranking Model** | `.cbm` / `.onnx` | 72 MB / MB | Веса градиентного бустинга |
| **User Embeddings** | `.npy` | 1.5M | 6040 пользователей × 64 dim |
| **Item Embeddings** | `.npy` | 927K | 3900 фильмов × 64 dim |

### 7.3 Скорость работы Retrieval (Поиск кандидатов)

* **Получение эмбеддинга (User Lookup):** `` (из кэша или In-memory).
* **Поиск ТОП-100 кандидатов (Annoy):** `` (на CPU, Annoy).
* **Итого Retrieval Latency:** ``.

---

## 8. Отчет по экспериментам оптимизации инференса

Этот раздел предназначен для фиксации результатов бенчмарков. Основная цель — уложиться в **P95 < 50ms** для полного цикла (Retrieval + Feature Lookup + Scoring).

### 8.1 Сравнение RPS и Latency

| ID | Описание Эксперимента | Инфраструктура (Docker) | RPS (Max) | Users | RPS (Saturated) | Latency (P95) Max Load ms | Вывод |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | Базовый запуск (как было) | 0.5 CPU, 2gb | 30 | 30 | 80 | 4500 | |
| 2 | Увеличение количества workers | 0.5 CPU, 2gb | | | | | красный горб. В Памяти теперь хранятся 4 копии катбуста, RPS упал, latency возрос, CPU забит на 50%, память прыгает от 50-90% то забивается то освобождается |
| 3 | Увеличение количества cpu до 1 | 1 CPU, 2gb | | | | | Красный горб, но прошел он гораздо быстрее. Почти линейный рост по сравнению с прошлым экспериментом. Я зафиксировал что на 1 воркер в unicorn фиксированно отходит 1 cpu. Ранее на каждый воркер было 1/8 CPU. теперь каждый воркер получает 1/4 CPU. |
| 4 | Увеличение количества cpu до 4 | 4 CPU, 2gb | 660 | 100 | 660 | 400 | Ошибки есть redis Slow, но в подавляющем большинстве это Total Slow. Основной вклад - катбуст - 189.06 мс в среднем |
| 5 | Добавил извлечение genres из ht | 4 CPU, 2gb | 911 | 90 | 143 | 1300 | 1 эксперимент по оптимизации самого приложения в коде |
| 6 | Перевод в X без pandas, чисто tuples | 4 CPU, 2gb | 925 | 90 | 518 | 700 | 2 эксперимент. Теперь qdrant начинает шалить |
| 7 | qdrant - перешел на grpc транспортировку векторов | 4 CPU, 2gb | 1188 | 100 | 566 | 800 | 1 эксперимент по оптимизации qdrant |
| 8 | qdrant - убрал вольюм из docker-compose | 4 CPU, 2gb | 1310 | 100 | 1310 | 3 | Win! |

### 8.2 Выводы по оптимизации

- сначала просто было много ошибок Total SLA
- потом (при увеличении числа workers, при фиксированно скудном cpu) столкнулся с проблемой того что появлялся "красный горб" из ошибок с самого начала, который потом нормализовывался.
- Затем устранял задержки внутри recsys-service - они заключались в постобработке данных (решил в основном добавлением создания хэш табл при инициализации)
- Поменял протокол общения с Qdrant с http на gRPC
- Финальный трюк с отключением volume - чисто прикол Docker-а, заключащийся в медленном сообщение между папкой и тем что в контейнере

---

## 9. Реализация (План действий)

### День 1: Оффлайн-часть

* [x] Подготовка данных (Polars).
* [x] Обучение CatBoostRanker (YetiRank).
* [x] Экспорт весов и эмбеддингов.

### День 2: Инфраструктура и Retrieval

* [x] Поднятие Redis и наполнение фичами (User/Item).
* [x] Реализация FAISS-сервиса для отбора кандидатов.
* [x] Написание FastAPI-эндпоинта `/recommend`.

### День 3: Оптимизация и Деплой

* [x] Проведение нагрузочного тестирования (Locust) и оптимизация Qdrant/Redis


