import numpy as np
import pickle
import polars as pl
import pandas as pd
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
from pathlib import Path
from loguru import logger
import sys

# --- КОНФИГУРАЦИЯ ---
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "movies"  # Имя коллекции
VECTOR_SIZE = 64            # Должно совпадать с тем, что было в ALS (factors=64)

# Пути (проверь, что файлы существуют)
VECTORS_PATH = "embeddings/item_vectors.npy"
MAPPINGS_PATH = "embeddings/mappings.pkl"
MOVIES_META_PATH = "data/movies.dat"

# Настройка логгера
logger.remove()
logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")

# 1. Подключение к Qdrant
client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)
try:
    client.get_collections()
    logger.success(f"Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
except Exception as e:
    logger.error(f"Could not connect to Qdrant: {e}")
    sys.exit(1)

# 2. Загрузка Артефактов (Вектора + Маппинги)
logger.info("Loading artifacts...")
item_vectors = np.load(VECTORS_PATH)

with open(MAPPINGS_PATH, "rb") as f:
    mappings = pickle.load(f)
    idx_to_item = mappings["idx_to_item"] # Internal Index -> Real MovieID

# 3. Загрузка Метаданных (для Payload)
# Чтобы в Qdrant лежали не только цифры, но и жанры/названия
logger.info("Loading movie metadata...")
# --- 2. ЗАГРУЗКА ФИЛЬМОВ (ITEMS) ---
logger.info("Reading Movies...")
movies_df = pd.read_csv(
    MOVIES_META_PATH,
    sep="::",
    engine="python",
    encoding="iso-8859-1",
    names=["MovieID", "Title", "Genres"],
)

movies_df = pl.from_pandas(movies_df)
# Превращаем в словарь для быстрого доступа: ID -> {Title: ..., Genres: ...}
movies_meta = {
    row["MovieID"]: {"title": row["Title"], "genres": row["Genres"]}
    for row in movies_df.to_dicts()
}

# 4. Пересоздание коллекции
# Если коллекция есть - удаляем и создаем заново (Idempotency)
logger.info(f"Recreating collection '{COLLECTION_NAME}'...")
client.recreate_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.DOT),
)

# 5. Подготовка точек (Points)
logger.info("Preparing points for upload...")
points = []

# Итерируемся по внутренним индексам матрицы (0...N)
for internal_idx, vector in enumerate(item_vectors):
    # Получаем реальный ID фильма
    real_movie_id = idx_to_item.get(internal_idx)
    
    if real_movie_id is None:
        continue
        
    # Достаем метаданные для пейлоада
    meta = movies_meta.get(real_movie_id, {"title": "Unknown", "genres": ""})
    
    # Создаем точку
    # Важно: Вектор нужно превратить в list (JSON serializable)
    points.append(models.PointStruct(
        id=real_movie_id,  # Используем Real ID!
        vector=vector.tolist(),
        payload=meta
    ))

# 6. Загрузка (Batch Upload)
# Qdrant любит батчи, но для 4k фильмов можно залить одним куском
logger.info(f"Uploading {len(points)} points...")

operation_info = client.upsert(
    collection_name=COLLECTION_NAME,
    wait=True,
    points=points
)

logger.success(f"🎉 Done! Status: {operation_info.status}")

# 7. Тестовый поиск (Sanity Check)
logger.info("Performing sanity check search (finding 'Toy Story' analogs)...")
# Ищем соседей для фильма с ID 1 (обычно Toy Story)
hits = client.retrieve(
    collection_name=COLLECTION_NAME,
    ids=[1],
    with_vectors=True # достать вектор, чтобы им поискать
)
