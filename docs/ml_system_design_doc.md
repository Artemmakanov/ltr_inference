# ML System Design Doc: Real-time Movie Re-ranker

## 1. Goal & Overview

**Business Goal:** Улучшить персонализацию выдачи фильмов для пользователей
**Technical Goal:** Создать Low-Latency сервис ранжирования (Learning-to-Rank), оптимизированный для работы в ограниченных ресурсах (CPU Inference), и продемонстрировать полный цикл MLOps: от подготовки данных до инференса.

## 2. Metrics & Constraints

### 2.1 Offline Metrics (Quality)

* **NDCG@10 (Normalized Discounted Cumulative Gain):** Основная метрика ранжирования. Показывает, насколько "правильные" фильмы находятся в топе выдачи.
* **ROC-AUC:** Вспомогательная метрика для оценки способности модели различать клик/не-клик (global ranking quality).

### 2.2 System Metrics (Performance)

* **Latency (P95):** < 50ms (на батч из 100 кандидатов). Критически важно для Real-time системы.
* **Model Size:** < 100 MB (для быстрой загрузки и деплоя).
* **Throughput:** > 100 RPS на одном инстансе (CPU).

### 2.3 Constraints

* **Hardware:** Training – GPU Cluster; Inference – CPU (Mac/Docker).
* **Data:** MovieLens 1M (Static snapshot).
* **Time:** 3-day sprint MVP.

## 3. Architecture

Система реализует классическую **Two-Stage Architecture**:

1. **Candidate Generation (Retrieval):**
* *Input:* User ID.
* *Logic:* Быстрый отбор ~100 релевантных кандидатов из тысяч.
* *Algorithm:* Faiss (ANN search) по эмбеддингам (ALS/SVD) или эвристика (Popularity / Genre Filter).


2. **Re-ranking (Scoring):**
* *Input:* User Features + List of Candidates.
* *Logic:* Пересчет скора для каждого кандидата с учетом сложных фичей.
* *Model:* **CatBoostRanker (YetiRank)** -> Export to **ONNX**.



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

### 5.1 Baseline

* **Top-Popular:** Ранжирование по среднему рейтингу / числу просмотров.
* *Ожидание:* Низкий NDCG, но мгновенный инференс.

### 5.2 Main Model (LTR)

* **Algorithm:** Gradient Boosting on Decision Trees (CatBoost).
* **Objective:** `YetiRank` (Listwise loss optimization). Это SOTA подход, который оптимизирует напрямую метрики ранжирования, а не ошибку классификации.
* **Parameters:**
* `group_id`: UserID.
* `depth`: 6-8.
* `task_type`: GPU (Training).



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



## 7. Implementation Plan (Sprint)

### Day 1: Training Pipeline (Offline)

* [x] Setup Repo Structure (`src/`, `configs/`).
* [x] ETL pipeline on Polars (Load -> Clean -> Split).
* [x] `CatBoostRanker` Training with CPU.
* [ ] Export artifacts: `ranker.cbm`, `user_embeddings.npy`.

### Day 2: Infrastructure & Retrieval

* [ ] Docker Compose setup (Redis, App).
* [ ] Feature Store ingestion script (Parquet -> Redis).
* [ ] Candidate Generation stub (Faiss or Genre-based heuristic).
* [ ] FastAPI skeleton.

### Day 3: Serving & Optimization

* [ ] Integration: Retrieval + Feature Lookup + Model.
* [ ] Optimization: Convert to ONNX.
* [ ] Benchmark: Locust load testing (Latency measurement).
* [ ] Gradio UI Demo.

## 8. Risks & Non-Goals

* **Out of Scope:** Real-time updates (обучение на новых кликах мгновенно). Модель обновляется раз в сутки (Batch Retraining).
* **Risk:** Холодный старт (новые юзеры). *Mitigation:* Фолбэк на Top-Popular.
* **Risk:** Сложность ONNX конвертации для категориальных фичей. *Mitigation:* Если ONNX упадет, используем нативный C++ инференс CatBoost.
