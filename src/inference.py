import polars as pl
from catboost import CatBoostRanker
import pandas as pd
from typing import List, Dict, Any
from .config import AppConfig

class ModelService:
    def __init__(self, config_path: str = "configs/base.yaml"):
        self.cfg = AppConfig.load(config_path)
        self.model = None
        self._items_features = None  # In-memory Feature Store (Items)
        self._titles_map = None      # ID -> Title
        
    def load(self):
        """Загружает модель и метаданные"""
        # 1. Загрузка модели
        self.model = CatBoostRanker()
        self.model.load_model(self.cfg.paths.model_output)
        
        # 2. Загрузка фичей айтемов (Genres) + Titles
        # Предполагаем, что movies.dat распарсен в parquet или лежит в raw
        # Для демо загрузим словарь: MovieID -> {Genres: ..., Title: ...}
        
        # Загружаем сырые данные для маппинга
        movies_df = pl.read_csv(
            "data/movies.dat", 
            separator=":", 
            has_header=False, 
            new_columns=["MovieID", "Title", "Genres"], 
            truncate_ragged_lines=True
        )
        
        # Preprocessing (как в features.py)
        movies_df = movies_df.with_columns(
            pl.col("Genres").cast(pl.String),
            pl.col("MovieID").cast(pl.Int64)
        )
        
        # Сохраняем для быстрого доступа
        self._items_df = movies_df
        self._titles_map = dict(zip(movies_df["MovieID"], movies_df["Title"]))
        
    def predict(self, user_features: Dict[str, Any], candidate_ids: List[int]) -> List[Dict]:
        """
        Ранжирует список кандидатов для конкретного профиля юзера.
        """
        if not candidate_ids:
            return []

        # 1. Формируем DataFrame кандидатов (Items)
        candidates_df = self._items_df.filter(pl.col("MovieID").is_in(candidate_ids))
        
        if candidates_df.is_empty():
            return []

        # 2. Добавляем фичи юзера (Cross Join: 1 юзер x N айтемов)
        # Важно: типы данных должны совпадать с тем, что было при обучении!
        user_df = pl.DataFrame([user_features])
        
        # Кастинг типов для юзера (как в features.py)
        for col in self.cfg.features.user_cat:
            if col in user_df.columns:
                user_df = user_df.with_columns(pl.col(col).cast(pl.String))
                
        # Создаем батч для инференса
        inference_batch = user_df.join(candidates_df, how="cross")
        
        # 3. Выбираем колонки в правильном порядке (как в конфиге)
        # CatBoost капризен к порядку, если подавать pd.DataFrame без Pool
        feature_cols = self.cfg.features.user_cat + self.cfg.features.item_cat + self.cfg.features.item_text
        
        X = inference_batch.select(feature_cols).to_pandas()
        
        # 4. Predict
        scores = self.model.predict(X)
        
        # 5. Формируем ответ
        results = []
        movie_ids = inference_batch["MovieID"].to_list()
        
        for mid, score in zip(movie_ids, scores):
            results.append({
                "movie_id": mid,
                "title": self._titles_map.get(mid, "Unknown"),
                "score": float(score),
                "genres": inference_batch.filter(pl.col("MovieID") == mid)["Genres"][0]
            })
            
        # Сортируем по скору (от большего к меньшему)
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return results

    def get_top_popular(self, k=50) -> List[int]:
        """Хелпер: возвращает ID самых популярных фильмов (для кандидатов)"""
        # В реальной жизни это делает Faiss/ANN. Тут просто заглушка.
        # Берем первые 50 ID для примера (или лучше загрузить ratings и посчитать top)
        return self._items_df["MovieID"].head(k).to_list()