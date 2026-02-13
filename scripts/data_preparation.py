from src.config import AppConfig
from src.features import RecSysFeaturePipeline
from pathlib import Path
import pandas as pd
import polars as pl

config = AppConfig.load("configs/base.yaml")

DATA_DIR = Path("./data")

ratings_raw = pd.read_csv(
    DATA_DIR / "ratings.dat",
    sep="::",
    engine="python",
    encoding="iso-8859-1",
    names=["UserID", "MovieID", "Rating", "Timestamp"],
)

ratings_raw = pl.from_pandas(ratings_raw)
movies_raw = pd.read_csv(
    DATA_DIR / "movies.dat",
    sep="::",
    engine="python",
    encoding="iso-8859-1",
    names=["MovieID", "Name", "Genres"],
)

movies_raw = pl.from_pandas(movies_raw)
users_raw = pd.read_csv(
    DATA_DIR / "users.dat",
    sep="::",
    engine="python",
    encoding="iso-8859-1",
    names=["UserID", "Gender", "Age", "Occupation", "Zip-code"]
)

users_raw = pl.from_pandas(users_raw)

# ... загрузка данных ...

pipeline = RecSysFeaturePipeline(config)

# 1. Препроцессинг
users_proc = pipeline.preprocess_users(users_raw)
items_proc = pipeline.preprocess_items(movies_raw)

# 2. Сборка полного датасета
full_dataset = pipeline.create_interaction_matrix(ratings_raw, users_proc, items_proc)

# 3. Сплит
train_df, test_df = pipeline.time_split(full_dataset)

# 4. Проверка на будущее (Data Leakage check)
# Убеждаемся, что минимальное время теста >= максимального времени трейна
assert test_df["Timestamp"].min() >= train_df["Timestamp"].max()

# 5. Сохранение для CatBoost (parquet быстрее csv)
train_df.write_parquet("data/train.parquet")
test_df.write_parquet("data/test.parquet")