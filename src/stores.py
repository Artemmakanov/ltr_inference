import polars as pl
import pandas as pd
import redis
from typing import List, Dict, Any, Protocol

class FeatureStore(Protocol):
    """Интерфейс, который мы будем соблюдать и для Redis, и для Memory"""
    def get_user_features(self, user_id: int) -> Dict[str, Any]:
        ...
    
    def get_item_features(self, item_ids: List[int]) -> pl.DataFrame:
        ...

class InMemoryFeatureStore:
    def __init__(self, users_path="data/users.dat", movies_path="data/movies.dat"):
        self.users_path = users_path
        self.movies_path = movies_path
        self._load_data()

    def _load_data(self):
        print("⏳ Loading In-Memory Feature Store...")
        
        # 1. Users (Парсим как в features.py)
        # Для демо упростим чтение (предполагаем, что файлы есть)

        self.users_df = pd.read_csv(
            self.users_path,
            sep="::",
            engine="python",
            encoding="iso-8859-1",
            names=["UserID", "Gender", "Age", "Occupation", "Zip-code"]
        )

        self.users_df = pl.from_pandas(self.users_df).with_columns(
            pl.col("UserID").cast(pl.Int32),
            pl.col("Gender").cast(pl.String).cast(pl.Categorical),
            pl.col("Age").cast(pl.Int32),
            pl.col("Occupation").cast(pl.String).cast(pl.Categorical),
            pl.col("Zip-code").cast(pl.String).cast(pl.Categorical),
        )
        
        # Превращаем в словарь для быстрого поиска по ID O(1)
        # {1: {"Gender": "F", "Age": 1}, ...}
        self.users_map = {
            row["UserID"]: {k: v for k, v in row.items() if k != "UserID"}
            for row in self.users_df.to_dicts()
        }

        # 2. Items
        self.items_df = pd.read_csv(
            "data/movies.dat",
            sep="::",
            engine="python",
            encoding="iso-8859-1",
            names=["MovieID", "Title", "Genres"],
        )

        self.items_df = pl.from_pandas(self.items_df).with_columns([
            pl.col("MovieID").cast(pl.Int32),
            pl.col("Genres").cast(pl.String).cast(pl.Categorical) # Важно для модели
        ])
        
        # Словарь для тайтлов (чтобы показывать в UI)
        self.titles_map = dict(zip(self.items_df["MovieID"], self.items_df["Title"]))
        
        print("✅ Feature Store loaded!")

    def get_user_features(self, user_id: int) -> Dict[str, Any]:
        # Эмуляция HGETALL user:{id}
        features = self.users_map.get(user_id)
        if not features:
            # Cold user fallback (возвращаем дефолт)
            return {"Gender": "M", "Age": 25, "Occupation": 0, "Zip-code": "10001"}
        
        # Кастинг типов для CatBoost (он любит строки)
        return {k: str(v) for k, v in features.items()}

    def get_item_features(self, item_ids: List[int]) -> pl.DataFrame:
        # Эмуляция MGET item:{id}
        # В Pandas/Polars это фильтрация
        subset = self.items_df.filter(pl.col("MovieID").is_in(item_ids))
        
        # Важно: Порядок строк может сбиться, но для джойна в модели это не страшно,
        # так как мы будем джойнить по ID.
        return subset

    def get_title(self, item_id: int) -> str:
        return self.titles_map.get(item_id, "Unknown")


class RedisFeatureStore:
    def __init__(self, host="localhost", port=6379):
        # decode_responses=True автоматически превращает bytes в str
        try:
            self.client = redis.Redis(host=host, port=port, decode_responses=True)
            self.client.ping() # Проверка соединения при старте
        except redis.ConnectionError:
            print(f"❌ WARNING: Could not connect to Redis at {host}:{port}")

    def get_user_features(self, user_id: int) -> Dict[str, Any]:
        """
        Возвращает фичи юзера.
        Если юзера нет (Cold Start), возвращает дефолтный профиль.
        """
        key = f"user:{user_id}"
        features = self.client.hgetall(key)
        
        if not features:
            # Fallback для холодных юзеров (средний профиль)
            # В реальном проде эти дефолты лучше вынести в конфиг
            return {
                "Gender": "M", 
                "Age": "25", 
                "Occupation": "0", 
                "Zip-code": "10001"
            }
        
        return features

    def get_item_features(self, item_ids: List[int]) -> pl.DataFrame:
        """
        Батчевая загрузка фичей для списка кандидатов.
        Использует Redis Pipeline для скорости.
        """
        if not item_ids:
            return pl.DataFrame()

        pipe = self.client.pipeline()
        for iid in item_ids:
            pipe.hgetall(f"item:{iid}")
        
        # Выполняем одним махом
        results = pipe.execute()
        
        # Собираем данные
        data = []
        for iid, props in zip(item_ids, results):
            if props: # Если айтем найден в Redis
                props["MovieID"] = iid
                data.append(props)
        
        if not data:
            return pl.DataFrame()

        # Создаем Polars DataFrame
        df = pl.DataFrame(data)
        
        # Приведение типов (Важно для CatBoost!)
        # MovieID в Int64, категориальные фичи в String
        # Если каких-то колонок нет, polars может ругаться, поэтому используем col(...).cast() аккуратно
        
        # Жанры точно нужны как строки
        if "Genres" in df.columns:
            df = df.with_columns(pl.col("Genres").cast(pl.String).cast(pl.Categorical))
            
        return df.with_columns(pl.col("MovieID").cast(pl.Int64))

    def get_title(self, item_id: int) -> str:
        """Быстрый метод для UI/Demo, чтобы получить название"""
        title = self.client.hget(f"item:{item_id}", "Title")
        return title if title else f"Unknown Movie ({item_id})"