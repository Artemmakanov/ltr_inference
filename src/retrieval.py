import annoy
import numpy as np
import pickle
from typing import List

class AnnoyRetriever:
    def __init__(self, 
                 user_vectors_path="embeddings/user_vectors.npy",
                 item_vectors_path="embeddings/item_vectors.npy",
                 mappings_path="embeddings/mappings.pkl"):
        
        self.user_vectors = np.load(user_vectors_path)
        self.item_vectors = np.load(item_vectors_path)
        
        with open(mappings_path, "rb") as f:
            self.mappings = pickle.load(f)
            
        self.dim = self.item_vectors.shape[1]
        self.index = annoy.AnnoyIndex(self.dim, 'dot') # Dot product для ALS
        
        self._build_index()

    def _build_index(self):
        print("⏳ Building Annoy Index...")
        # Добавляем все вектора айтемов в индекс
        # Важно: Annoy работает с integer ID (0...N). 
        # Мы используем наши внутренние индексы из item_to_idx.
        for i in range(self.item_vectors.shape[0]):
            self.index.add_item(i, self.item_vectors[i])
            
        self.index.build(10) # 10 деревьев - баланс точность/скорость
        print("✅ Annoy Index built!")

    def get_candidates(self, user_id: int, k=100) -> List[int]:
        """
        Возвращает список Real MovieIDs
        """
        # 1. Получаем внутренний индекс юзера
        u_idx = self.mappings["user_to_idx"].get(user_id)
        
        if u_idx is None:
            # Cold Start: Если юзера нет, возвращаем пустой список 
            # (в реале тут фоллбэк на топ популярных)
            print(f"User {user_id} not found in embeddings.")
            return []
        
        # 2. Берем вектор юзера
        query_vector = self.user_vectors[u_idx]
        
        # 3. Ищем соседей (возвращает внутренние индексы айтемов)
        neighbor_indices = self.index.get_nns_by_vector(query_vector, k)
        
        # 4. Конвертируем обратно в Real MovieID
        candidates = [self.mappings["idx_to_item"][idx] for idx in neighbor_indices]
        
        return candidates