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

* **Hardware:** Training – CPU; Inference – CPU (Mac/Docker).
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

* **HIT@5:** `0.2204` (Доля случаев, когда релевантный фильм попал в топ-5 рекомендаций).
* **Время обучения (CPU):** `201 сек` (CatBoost YetiRank, 1M транзакций, глубина 6).

### 7.2 Характеристики артефактов

| Артефакт | Формат | Размер | Описание |
| --- | --- | --- | --- |
| **Ranking Model** | `.cbm` / `.onnx` | ~20 MB | Веса градиентного бустинга |
| **User Embeddings** | `.npy` | ~15 MB | 6040 пользователей × 64 dim |
| **Item Embeddings** | `.npy` | ~2 MB | 3900 фильмов × 64 dim |

### 7.3 Скорость работы Retrieval (Поиск кандидатов)

* **Получение эмбеддинга (User Lookup):** `` (из кэша или In-memory).
* **Поиск ТОП-100 кандидатов (Annoy):** `` (на CPU, Annoy).
* **Итого Retrieval Latency:** ``.

---

## 8. Отчет по экспериментам оптимизации инференса

Этот раздел предназначен для фиксации результатов бенчмарков. Основная цель — уложиться в **P95 < 50ms** для полного цикла (Retrieval + Feature Lookup + Scoring).

### 8.1 Сравнение Latency (Batch: 100 кандидатов)
<!-- 
| Метод инференса | P50 (ms) | P95 (ms) | RPS | Комментарий |
| --- | --- | --- | --- | --- |
| **Native CatBoost (.predict)** |  |  |  | Высокий overhead Python-обертки |
| **CatBoost + ONNX Runtime** |  |  |  | Оптимизированный граф вычислений |
| **Redis MGET (Feature Lookup)** |  |  |  | Время на десериализацию фичей |
| **End-to-End Pipeline** |  |  |  | Полный путь запроса | -->

### 8.2 Выводы по оптимизации

<!-- * **Векторизация vs Циклы:** Использование `Polars` или `NumPy` для формирования входной матрицы вместо Python `dict` сократило Latency на [X]%.
* **Сжатие данных:** Использование `MessagePack` вместо `JSON` для хранения фичей в Redis уменьшило сетевой трафик и время парсинга на [Y]%.
* **ONNX vs Native:** Переход на ONNX позволил/не позволил (нужное подчеркнуть) выиграть в стабильности задержек (снижение jitter). -->

---

## 9. Реализация (План действий)

### День 1: Оффлайн-часть (Ready)

* [x] Подготовка данных (Polars).
* [x] Обучение CatBoostRanker (YetiRank).
* [x] Экспорт весов и эмбеддингов.

### День 2: Инфраструктура и Retrieval (In Progress)

* [ ] Поднятие Redis и наполнение фичами (User/Item).
* [ ] Реализация FAISS-сервиса для отбора кандидатов.
* [ ] Написание FastAPI-эндпоинта `/recommend`.

### День 3: Оптимизация и Деплой

* [ ] Конвертация модели в ONNX.
* [ ] Проведение нагрузочного тестирования (Locust).
* [ ] Подготовка финального отчета в Gradio.

