import sys
import uvicorn
from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
from loguru import logger
import time

# Импортируем наши модули
from src.inference import ModelService
from src.stores import RedisFeatureStore
from src.retrieval import QdrantRetriever

# Настройка логгера
logger.remove()
logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")

# --- 1. Data Models (Pydantic) ---
# Описываем, что входит и что выходит (валидация + документация Swagger)

class RecommendRequest(BaseModel):
    user_id: int
    top_k: int = 10

class MovieResponse(BaseModel):
    movie_id: int
    title: str
    score: float
    genres: str

class RecommendResponse(BaseModel):
    user_id: int
    candidates_found: int
    recommendations: List[MovieResponse]
    inference_time_ms: float

# --- 2. Global Services (Singleton Pattern) ---
# Инициализируем один раз при старте, чтобы не пересоздавать соединения

app = FastAPI(title="RecSys Movie Ranker", version="1.0.0")

class ServiceContainer:
    def __init__(self):
        logger.info("🔥 [1/4] Initializing Services Container...")
        
        # 1. Feature Store (Redis)
        logger.info("   ... Connecting to Redis")
        # Добавляем socket_timeout, чтобы не висело вечно!
        self.features = RedisFeatureStore(host="localhost", port=6379)
        # Если RedisFeatureStore делает ping внутри __init__, он может зависнуть там
        
        logger.info("🔥 [2/4] Redis Connected (or skipped)")

        # 2. Retrieval (Qdrant)
        logger.info("   ... Connecting to Qdrant")
        self.retriever = QdrantRetriever(host="localhost", port=6333)
        
        logger.info("🔥 [3/4] Qdrant Connected")

        # 3. Ranking Model (CatBoost)
        logger.info("   ... Loading CatBoost Model")
        self.ranker = ModelService()
        self.ranker.load() 
        
        logger.info("🔥 [4/4] Model Loaded. Service Ready!")

# Глобальная переменная для сервисов
services: ServiceContainer = None

@app.on_event("startup")
def startup_event():
    global services
    services = ServiceContainer()

# --- 3. API Endpoints ---

@app.get("/health")
def health_check():
    """Проверка, что сервис жив"""
    return {"status": "ok", "model_loaded": services.ranker.model is not None}

@app.post("/recommend", response_model=RecommendResponse)
def get_recommendations(request: RecommendRequest):
    """
    Главная ручка: Retrieval -> Feature Fetching -> Ranking
    """
    logger.info("DEBUG: Request received")  # 1
    start_time = time.time()
    user_id = request.user_id
    
    try:
        # A. Candidate Generation (Retrieval)
        # -----------------------------------
        # Ищем 100 кандидатов в Qdrant
        logger.info("DEBUG: Calling Qdrant...") # 2
        candidate_ids = services.retriever.get_candidates(user_id, k=100)
        logger.info(f"DEBUG: Qdrant done. Found: {len(candidate_ids)}") # 3
        
        # Fallback: Если Qdrant вернул пустоту (Cold User), берем популярное
        if not candidate_ids:
            logger.warning(f"User {user_id} is cold (no embeddings). Using Popular fallback.")
            candidate_ids = services.ranker.get_top_popular(k=100)
            
        # B. Feature Fetching (Store)
        # ---------------------------
        # Получаем фичи юзера из Redis
        logger.info("DEBUG: Calling Redis...") # 4
        user_features = services.features.get_user_features(user_id)
        logger.info("DEBUG: Redis done") # 5
        
        # C. Ranking (Inference)
        # ----------------------
        # Сервис сам достанет фичи айтемов и прогонит CatBoost
        logger.info("DEBUG: Calling CatBoost...") # 6
        ranked_items = services.ranker.predict(user_features, candidate_ids)
        logger.info("DEBUG: CatBoost done") # 7
        # D. Response Formatting
        # ----------------------
        # Берем топ-K
        top_items = ranked_items[:request.top_k]
        
        response_list = [
            MovieResponse(
                movie_id=item["movie_id"],
                title=item["title"],
                score=item["score"],
                genres=item["genres"]
            )
            for item in top_items
        ]
        
        execution_time = (time.time() - start_time) * 1000
        
        logger.info(f"Recs for User {user_id}: Found {len(candidate_ids)} candidates, returned {len(top_items)}. Time: {execution_time:.2f}ms")
        
        return RecommendResponse(
            user_id=user_id,
            candidates_found=len(candidate_ids),
            recommendations=response_list,
            inference_time_ms=execution_time
        )

    except Exception as e:
        logger.error(f"Error processing request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Запуск сервера
    uvicorn.run(app, host="0.0.0.0", port=8001)