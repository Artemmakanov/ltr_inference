import sys
from loguru import logger
from catboost import CatBoostRanker
from src.config import AppConfig
from src.dataset import LTRDataset

# Настройка логгера
logger.remove()
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")

# 1. Загрузка конфигурации
# Если хочешь передать другой конфиг, можно брать из sys.argv
config_path = "configs/base.yaml" 
logger.info(f"Loading config from {config_path}...")
cfg = AppConfig.load(config_path)

# 2. Подготовка данных
# LTRDataset сам отсортирует по UserID и создаст Pool
loader = LTRDataset(cfg)

logger.info("Loading TRAIN pool...")
train_pool = loader.load_pool(cfg.paths.train_data)

logger.info("Loading TEST pool...")
test_pool = loader.load_pool(cfg.paths.test_data)

# 3. Инициализация модели
logger.info(f"Initializing CatBoostRanker (Task: {cfg.catboost.task_type}, Loss: {cfg.catboost.loss_function})...")

model = CatBoostRanker(
    iterations=cfg.catboost.iterations,
    learning_rate=cfg.catboost.learning_rate,
    depth=cfg.catboost.depth,
    loss_function=cfg.catboost.loss_function,
    task_type=cfg.catboost.task_type,
    verbose=cfg.catboost.verbose,
    # Добавляем метрики для мониторинга. 
    # В CatBoost для бинарных целей RecallAt:top=K эквивалентен Hit@K
    custom_metric=['RecallAt:top=5', 'PFound', 'NDCG:top=5'],
    eval_metric='NDCG:top=5' # Используем как основную для Early Stopping
)

model.fit(
    train_pool,
    eval_set=test_pool,
    plot=False,
    use_best_model=True
)

# 5. Извлечение Hit@5 (RecallAt:top=5)
final_metrics = model.get_best_score()

# Структура: {'validation': {'RecallAt:top=5': 0.123, 'NDCG:top=5': 0.456}}
hit5 = final_metrics['validation']['RecallAt:top=5']

logger.success(f"Final Hit@5 (via RecallAt) on test pool: {hit5:.4f}")

# 5. Сохранение
output_path = cfg.paths.model_output
logger.info(f"Saving model to {output_path}...")

# Сохраняем в родном формате CatBoost
model.save_model(output_path)

# Можно сразу экспортировать в ONNX, если нужно (раскомментируй для теста)
# onnx_path = output_path.replace(".cbm", ".onnx")
# model.save_model(onnx_path, format="onnx", export_parameters={'onnx_domain': 'ai.catboost'})
# logger.success(f"Model exported to ONNX: {onnx_path}")

logger.success("Training pipeline finished successfully!")

