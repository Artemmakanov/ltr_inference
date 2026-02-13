import polars as pl
from catboost import Pool
from typing import Optional, Tuple
from loguru import logger
from src.config import AppConfig

class LTRDataset:
    """
    Класс-обертка для загрузки данных и конвертации в CatBoost Pool.
    Специально для задач Learning-to-Rank.
    """
    def __init__(self, config: AppConfig):
        self.cfg = config

    def load_pool(self, data_path: str, is_train: bool = True) -> Pool:
        """
        Загружает parquet, сортирует по UserID (важно для ранжирования!) 
        и возвращает готовый Pool.
        """
        logger.info(f"Loading data from {data_path}...")
        
        # 1. Читаем Parquet через Polars (лениво или сразу)
        df = pl.read_parquet(data_path)
        
        # 2. КРИТИЧНО ДЛЯ РАНЖИРОВАНИЯ:
        # Данные должны быть сгруппированы по GroupID (UserID).
        # CatBoost ожидает, что все айтемы одного юзера идут подряд.
        logger.info("Sorting by UserID (Group ID) for LTR...")
        df = df.sort("UserID")

        # 3. Выделяем X (фичи), y (таргет) и группы
        X = df.select(
            self.cfg.features.user_num + 
            self.cfg.features.user_cat + 
            self.cfg.features.item_cat
        )
        y = df[self.cfg.features.target_col]
        
        # Группы (Query ID) - нужны для YetiRank/QuerySoftmax
        groups = df["UserID"]

        # 4. Собираем Pool
        logger.info(f"Creating CatBoost Pool with {len(df)} rows...")
        
        pool = Pool(
            data=X.to_pandas(), # CatBoost пока лучше работает с pandas/numpy внутри
            label=y.to_pandas(),
            group_id=groups.to_pandas(), # Указываем группы!
            
            # Передаем имена колонок, чтобы CatBoost сам понял индексы
            cat_features=self.cfg.features.user_cat + self.cfg.features.item_cat,
        )
        
        return pool