import polars as pl
from typing import List, Tuple, Dict, Union
from dataclasses import dataclass
from loguru import logger
import sys

from .config import AppConfig

# Настройка loguru
logger.remove()
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")

class RecSysFeaturePipeline:
    def __init__(self, config: AppConfig):
        self.config = config
        
    def preprocess_users(self, users_df: pl.DataFrame) -> pl.DataFrame:
        logger.info("Preprocessing Users...")
        exprs = [
            pl.col(col).cast(pl.String).cast(pl.Categorical).alias(col) 
            for col in self.config.features.user_cat
            if col in users_df.columns
        ]

        exprs += [
            pl.col(col).cast(pl.Float64).alias(col) 
            for col in self.config.features.user_num
            if col in users_df.columns
        ]
        
        if "UserID" not in [e.name for e in exprs]:
             exprs.append(pl.col("UserID"))
             
        return users_df.select(exprs)

    def preprocess_items(self, items_df: pl.DataFrame) -> pl.DataFrame:
        logger.info("Preprocessing Items...")
        exprs = []
        
        # Обработка категориальных
        for col in self.config.features.item_cat:
            if col in items_df.columns:
                exprs.append(pl.col(col).cast(pl.String).cast(pl.Categorical))
        
        # Обработка текстовых (жанры)
        for col in self.config.features.item_text:
            if col in items_df.columns:
                exprs.append(pl.col(col).cast(pl.String))
                
        exprs.append(pl.col("MovieID"))
        return items_df.select(exprs)

    def create_interaction_matrix(self, 
                                ratings_df: pl.DataFrame, 
                                users_processed: pl.DataFrame, 
                                items_processed: pl.DataFrame) -> pl.DataFrame:
        """
        Создает полный датасет взаимодействий с фичами.
        """
        logger.info("Creating Interaction Matrix (Merging)...")
        
        # 1. Формируем таргет
        target_expr = (pl.col("Rating") > self.config.features.positive_threshold).cast(pl.Int8).alias(self.config.features.target_col)
        
        # Обязательно сохраняем Timestamp для сплита!
        interactions = ratings_df.select([
            pl.col("UserID"),
            pl.col("MovieID"),
            target_expr,
            pl.col("Timestamp") 
        ])

        # 2. Джойним фичи
        dataset = interactions.join(users_processed, on="UserID", how="left")
        dataset = dataset.join(items_processed, on="MovieID", how="left")
        
        # Удаляем строки с Null
        initial_len = len(dataset)
        dataset = dataset.drop_nulls()
        if len(dataset) < initial_len:
            logger.warning(f"Dropped {initial_len - len(dataset)} rows due to missing features")

        return dataset

    def filter_valid_groups(self, dataset: pl.DataFrame) -> pl.DataFrame:
        """
        [NEW] Очистка групп (UserIDs) для Listwise обучения (YetiRank).
        Удаляет юзеров, у которых:
        1. Только 1 запись (нечего сортировать).
        2. Все таргеты одинаковые (только 0 или только 1) - лосс будет нулевым.
        """
        logger.info("Filtering valid groups for Ranking (removing trivial groups)...")
        initial_users = dataset["UserID"].n_unique()
        target_col = self.config.features.target_col

        # Агрегируем статистику по каждой группе
        group_stats = (
            dataset.group_by("UserID")
            .agg([
                pl.col(target_col).min().alias("min_t"),
                pl.col(target_col).max().alias("max_t"),
                pl.col(target_col).count().alias("cnt")
            ])
        )

        # Оставляем только "интересные" группы
        valid_users = group_stats.filter(
            (pl.col("min_t") != pl.col("max_t")) &  # Есть и позитив, и негатив
            (pl.col("cnt") > 1)                     # Длина списка > 1
        ).select("UserID")

        # Фильтруем исходный датасет через Inner Join
        cleaned_dataset = dataset.join(valid_users, on="UserID", how="inner")
        
        final_users = cleaned_dataset["UserID"].n_unique()
        dropped_users = initial_users - final_users
        
        if dropped_users > 0:
            logger.warning(f"Dropped {dropped_users} users ({dropped_users/initial_users:.1%}) unsuitable for ranking.")
            logger.info(f"Final dataset size: {len(cleaned_dataset)} rows, {final_users} users.")
        else:
            logger.success("All groups are valid for ranking!")

        return cleaned_dataset

    def time_split(self, dataset: pl.DataFrame) -> Tuple[pl.DataFrame, pl.DataFrame]:
        """
        Глобальное разбиение по времени.
        """
        logger.info(f"Splitting dataset by time (Ratio: {self.config.features.train_ratio})...")
        
        dataset = dataset.sort("Timestamp")
        
        # Берем временную отсечку
        split_point = dataset["Timestamp"].quantile(self.config.features.train_ratio)
        
        logger.debug(f"Split timestamp threshold: {split_point}")
        
        train_df = dataset.filter(pl.col("Timestamp") < split_point)
        test_df = dataset.filter(pl.col("Timestamp") >= split_point)
        
        logger.success(f"Train size: {len(train_df)}, Test size: {len(test_df)}")
        
        return train_df, test_df