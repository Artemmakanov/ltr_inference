import polars as pl
import pandas as pd
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