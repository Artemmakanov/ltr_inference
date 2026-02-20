import sys
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional, Dict
from loguru import logger
import time

# Импортируем наши модули
from src.inference import ModelService
from src.stores import RedisFeatureStore
from src.retrieval import QdrantRetriever

# --- Настройка логгера ---
logger.remove()
logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")

# --- 1. Data Models (Pydantic) ---

class RecommendRequest(BaseModel):
    user_id: int
    top_k: int = 10

class MovieResponse(BaseModel):
    movie_id: int
    title: str
    score: float
    genres: str

# Добавили модель для детализации таймингов
class LatencyBreakdown(BaseModel):
    retrieval_ms: float  # Qdrant
    features_ms: float   # Redis
    ranking_ms: float    # CatBoost
    total_ms: float

class RecommendResponse(BaseModel):
    user_id: int
    candidates_found: int
    recommendations: List[MovieResponse]
    latency: LatencyBreakdown  # Вставляем сюда нашу детализацию

# --- 2. Global Services ---

app = FastAPI(title="RecSys Movie Ranker", version="1.1.0")

class ServiceContainer:
    def __init__(self):
        logger.info("🔥 [Init] Loading Services...")
        try:
            self.features = RedisFeatureStore()
            logger.info("   ✅ Redis Connected")
        except Exception as e:
            logger.error(f"   ❌ Redis Failed: {e}")
            raise e

        try:
            self.retriever = QdrantRetriever()
            logger.info("   ✅ Qdrant Connected")
        except Exception as e:
            logger.error(f"   ❌ Qdrant Failed: {e}")
            # Можно не рейзить ошибку, если допустимо работать без Qdrant
            raise e

        self.ranker = ModelService()
        self.ranker.load()
        logger.info("   ✅ CatBoost Loaded")

services: ServiceContainer = None

@app.on_event("startup")
def startup_event():
    global services
    services = ServiceContainer()

# --- 3. API Endpoints ---

@app.post("/recommend", response_model=RecommendResponse)
def get_recommendations(request: RecommendRequest, response: Response):
    """
    Pipeline: Retrieval (Qdrant) -> Features (Redis) -> Ranking (CatBoost)
    """
    t_start = time.time()
    user_id = request.user_id
    
    # Переменные для таймингов
    t_retrieval = 0.0
    t_features = 0.0
    t_ranking = 0.0
    
    candidates_ids = []
    
    # --- STEP 1: Retrieval (Qdrant) ---
    t0 = time.time()
    try:
        # Пытаемся достать кандидатов
        candidate_ids = services.retriever.get_candidates(user_id, k=100)
    except Exception as e:
        # Graceful Degradation: Если Qdrant упал, не крашим сервис, а пишем лог
        logger.error(f"⚠️ Retrieval Error (Qdrant): {e}. Switching to fallback.")
        candidate_ids = []
    
    t1 = time.time()
    t_retrieval = (t1 - t0) * 1000

    # Fallback Logic
    if not candidate_ids:
        logger.warning(f"User {user_id}: Cold or Qdrant failed. Using Popular.")
        # Предполагаем, что get_top_popular очень быстрый (из памяти)
        candidate_ids = services.ranker.get_top_popular(k=100)

    # --- STEP 2: Feature Store (Redis) ---
    # Это критическая секция. Если нет фичей юзера, мы не можем предсказать.
    t2_start = time.time()
    try:
        user_features = services.features.get_user_features(user_id)
        
        # Валидация данных: Проверяем, что фичи реально пришли
        if user_features is None:
             raise ValueError(f"Features for user {user_id} not found in Redis")
             
    except Exception as e:
        # Тут мы падаем честно, так как без фичей модель не работает
        logger.critical(f"❌ Feature Store Error (Redis): {e}")
        raise HTTPException(status_code=503, detail="Feature Store Unavailable or User Unknown")
        
    t2_end = time.time()
    t_features = (t2_end - t2_start) * 1000

    # --- STEP 3: Ranking (CatBoost) ---
    t3_start = time.time()
    try:
        ranked_items = services.ranker.predict(user_features, candidate_ids)
    except Exception as e:
        logger.critical(f"❌ Model Inference Error: {e}")
        raise HTTPException(status_code=500, detail="Ranking Model Failed")
        
    t3_end = time.time()
    t_ranking = (t3_end - t3_start) * 1000

    # --- Formatting Response ---
    top_items = ranked_items[:request.top_k]
    response_list = [
        MovieResponse(**item) for item in top_items
    ]
    
    t_total = (time.time() - t_start) * 1000
    
    # Логируем красивую разбивку
    logger.info(
        f"User {user_id} | Total: {int(t_total)}ms | "
        f"Qdrant: {int(t_retrieval)}ms | "
        f"Redis: {int(t_features)}ms | "
        f"Model: {int(t_ranking)}ms"
    )

    # --- ВАЖНО: Добавляем заголовки для Locust ---
    # Locust может читать заголовки ответа и строить по ним графики
    response.headers["X-Latency-Total"] = str(t_total)
    response.headers["X-Latency-Retrieval"] = str(t_retrieval)
    response.headers["X-Latency-Features"] = str(t_features)
    response.headers["X-Latency-Ranking"] = str(t_ranking)

    return RecommendResponse(
        user_id=user_id,
        candidates_found=len(candidate_ids),
        recommendations=response_list,
        latency=LatencyBreakdown(
            retrieval_ms=t_retrieval,
            features_ms=t_features,
            ranking_ms=t_ranking,
            total_ms=t_total
        )
    )

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8001, reload=False)