import polars as pl
import numpy as np
import pickle
import pandas as pd
from pathlib import Path
from loguru import logger

from scipy.sparse import csr_matrix

import implicit

from src.config import AppConfig

# Настройка путей
DATA_PATH = Path("data")
config = AppConfig.load("configs/base.yaml")
OUTPUT_DIR = Path(config.paths.embeddings_output)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Параметры ALS
FACTORS = 64        # Размерность эмбеддинга (обычно 32-128 для RecSys)
ITERATIONS = 15     # Количество эпох
REGULARIZATION = 0.1

logger.info("Loading ratings...")
# Читаем данные (подстрой под свой формат, если у тебя parquet)
# Для .dat (MovieLens)


# 1. Создание маппингов (Real ID -> Matrix Index)
# ALS работает с матрицей, где индексы - это целые числа от 0 до N-1.
# Нам нужно запомнить, какой Real UserID соответствует строке 0, 1, 2...

df = pd.read_csv(
    DATA_PATH / "ratings.dat",
    sep="::",
    engine="python",
    encoding="iso-8859-1",
    names=["UserID", "MovieID", "Rating", "Timestamp"],
)

df = pl.from_pandas(df)

logger.info("Creating ID mappings...")
unique_users = df["UserID"].unique().sort()
unique_items = df["MovieID"].unique().sort()

# Словари: RealID -> Index
user_to_idx = {uid: i for i, uid in enumerate(unique_users.to_list())}
item_to_idx = {iid: i for i, iid in enumerate(unique_items.to_list())}

# Обратные словари: Index -> RealID (понадобятся при выдаче рекомендаций)
idx_to_user = {i: uid for uid, i in user_to_idx.items()}
idx_to_item = {i: iid for iid, i in item_to_idx.items()}

# 2. Подготовка разреженной матрицы (CSR)
logger.info("Building sparse matrix...")

# Заменяем реальные ID на индексы в DataFrame
# map_dict в polars работает быстро
users_mapped = df["UserID"].replace(user_to_idx, default=None).cast(pl.Int32)
items_mapped = df["MovieID"].replace(item_to_idx, default=None).cast(pl.Int32)
ratings = df["Rating"].cast(pl.Float32)

# Создаем матрицу: Строки=Юзеры, Колонки=Айтемы
user_item_matrix = csr_matrix(
    (ratings.to_numpy(), (users_mapped.to_numpy(), items_mapped.to_numpy())),
    shape=(len(unique_users), len(unique_items))
)

# 3. Обучение ALS
logger.info(f"Training ALS model (Factors: {FACTORS})...")

# Инициализация модели
model = implicit.als.AlternatingLeastSquares(
    factors=FACTORS,
    regularization=REGULARIZATION,
    iterations=ITERATIONS,
    calculate_training_loss=True,
    random_state=42
)

# Обучение (implicit ожидает user_items matrix)
model.fit(user_item_matrix)

# 4. Сохранение артефактов
logger.info("Saving embeddings and mappings...")

# Эмбеддинги юзеров и айтемов (numpy array)
# model.user_factors и model.item_factors - это массивы (N_users, Factors)
user_vectors = model.user_factors
item_vectors = model.item_factors

# Обязательно кастим во float32 для Faiss/ONNX
np.save(OUTPUT_DIR / "user_vectors.npy", user_vectors.astype(np.float32))
np.save(OUTPUT_DIR / "item_vectors.npy", item_vectors.astype(np.float32))

# Сохраняем маппинги, чтобы сервис знал, кто есть кто
with open(OUTPUT_DIR / "mappings.pkl", "wb") as f:
    pickle.dump({
        "user_to_idx": user_to_idx,
        "item_to_idx": item_to_idx,
        "idx_to_user": idx_to_user,
        "idx_to_item": idx_to_item
    }, f)

logger.success(f"Done! Artifacts saved to {OUTPUT_DIR}")
logger.info(f"User vectors shape: {user_vectors.shape}")
logger.info(f"Item vectors shape: {item_vectors.shape}")
