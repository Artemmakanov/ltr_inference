import polars as pl
import redis
import sys
import pandas as pd
from loguru import logger
from pathlib import Path

# --- КОНФИГУРАЦИЯ ---
REDIS_HOST = "localhost"
REDIS_PORT = 6379
BATCH_SIZE = 1000  # Размер пачки для пайплайна

# Пути к файлам (укажи свои, если отличаются)
USERS_PATH = "data/users.dat"
MOVIES_PATH = "data/movies.dat"

# Настройка логгера
logger.remove()
logger.add(sys.stderr, format="<green>{time:HH:mm:ss}</green> | <level>{message}</level>")

def get_redis_client():
    try:
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
        r.ping()
        logger.success(f"Connected to Redis at {REDIS_HOST}:{REDIS_PORT}")
        return r
    except redis.ConnectionError:
        logger.error("Could not connect to Redis. Is Docker running?")
        sys.exit(1)

def ingest_data(r: redis.Redis, df: pl.DataFrame, key_prefix: str, id_col: str):
    """
    Универсальная функция заливки данных через Pipeline.
    """
    logger.info(f"Start ingesting {key_prefix} ({len(df)} rows)...")
    
    pipe = r.pipeline()
    count = 0
    
    # Конвертируем в список словарей для итерации (Polars -> Python Dicts)
    # Это быстро для таких объемов
    rows = df.to_dicts()
    
    for row in rows:
        # 1. Формируем ключ: user:123 или item:555
        # Важно: ID должен быть строкой или int
        entity_id = row.pop(id_col) 
        redis_key = f"{key_prefix}:{entity_id}"
        
        # 2. Чистим данные (None -> пустая строка, числа -> строки)
        # Redis хранит всё как bytes или string
        clean_row = {k: str(v) for k, v in row.items() if v is not None}
        
        # 3. Добавляем команду в пайплайн
        if clean_row:
            pipe.hset(redis_key, mapping=clean_row)
            count += 1
        
        # 4. Выполняем пачками (чтобы не забить память клиента)
        if count % BATCH_SIZE == 0:
            pipe.execute()
            
    # Доливаем остатки
    pipe.execute()
    logger.success(f"✅ Uploaded {count} keys for prefix '{key_prefix}'")

r = get_redis_client()

# --- 1. ЗАГРУЗКА ЮЗЕРОВ ---
logger.info("Reading Users...")
# Парсим ML-1M формат
users_df = pd.read_csv(
    USERS_PATH,
    sep="::",
    engine="python",
    encoding="iso-8859-1",
    names=["UserID", "Gender", "Age", "Occupation", "Zip-code"],
)

users_df = pl.from_pandas(users_df)

# Заливаем
ingest_data(r, users_df, key_prefix="user", id_col="UserID")

# --- 2. ЗАГРУЗКА ФИЛЬМОВ (ITEMS) ---
logger.info("Reading Movies...")
movies_df = pd.read_csv(
    MOVIES_PATH,
    sep="::",
    engine="python",
    encoding="iso-8859-1",
    names=["MovieID", "Title", "Genres"],
)

movies_df = pl.from_pandas(movies_df)

# Чуть препроцессинга, если надо (например, разбить жанры)
# Но для Feature Store храним "как есть" или чуть чистим
movies_df = movies_df.with_columns(
    pl.col("Genres").cast(pl.String)
)

# Заливаем
ingest_data(r, movies_df, key_prefix="item", id_col="MovieID")

logger.success("🎉 All data loaded into Redis!")
