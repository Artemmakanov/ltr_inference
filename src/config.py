from dataclasses import dataclass, field
from typing import List, Optional
from omegaconf import OmegaConf


@dataclass
class PathsConfig:
    train_data: str
    test_data: str
    model_output: str

@dataclass
class FeatureConfig:
    user_cat: List[str]
    user_num: List[str]
    item_cat: List[str]
    target_col: str = "target"
    positive_threshold: int = 3
    train_ratio: float = 0.8

@dataclass
class CatBoostConfig:
    iterations: int = 1000
    learning_rate: float = 0.03
    depth: int = 6
    loss_function: str = "YetiRank"
    task_type: str = "CPU"
    verbose: int = 100
    
    # Сюда можно добавить специфичные параметры для RecSys
    # group_id: str = "UserID" # Обычно для YetiRank нужен GroupID

@dataclass
class AppConfig:
    features: FeatureConfig
    catboost: CatBoostConfig
    paths: PathsConfig
    @classmethod
    def load(cls, config_path: str = "configs/base.yaml") -> "AppConfig":
        """
        Загружает YAML и валидирует его через типы датаклассов.
        """
        # 1. Загружаем структуру (схему) с дефолтными значениями
        schema = OmegaConf.structured(cls)
        
        # 2. Загружаем файл
        conf = OmegaConf.load(config_path)
        
        # 3. Мержим: файл переписывает дефолтные значения схемы
        merged_conf = OmegaConf.merge(schema, conf)
        
        # 4. Возвращаем типизированный объект (превращаем DictConfig обратно в Dataclass)
        # Это даст автокомплит в IDE!
        return OmegaConf.to_object(merged_conf)