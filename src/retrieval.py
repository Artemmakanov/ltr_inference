import annoy
import os
import numpy as np
import pickle
from typing import List, Optional
from qdrant_client import QdrantClient
# Импортируем модели для низкоуровневых запросов
from qdrant_client.http import models as rest_models

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
    


class QdrantRetriever:
    def __init__(self, 
                 collection_name="movies",
                 user_vectors_path="embeddings/user_vectors.npy",
                 mappings_path="embeddings/mappings.pkl"):
        
        self.host = os.getenv("QDRANT_HOST", "localhost")
        self.port = int(os.getenv("QDRANT_PORT", 6333))
        self.collection_name = collection_name
        
        # 1. Подключение
        self.client = QdrantClient(host=self.host, port=self.port)
        
        # 2. Загрузка векторов ЮЗЕРОВ
        print(f"⏳ Loading user vectors from {user_vectors_path}...")
        self.user_vectors = np.load(user_vectors_path)
        
        with open(mappings_path, "rb") as f:
            self.mappings = pickle.load(f)
            
        print("✅ QdrantRetriever initialized (Low-Level Mode)")

    def get_candidates(self, user_id: int, k=100) -> List[int]:
        """
        Возвращает список Real MovieIDs.
        """
        # 1. Получаем индекс юзера
        u_idx = self.mappings["user_to_idx"].get(user_id)
        
        if u_idx is None:
            print(f"⚠️ User {user_id} not found in embeddings.")
            return []
        
        # 2. Берем вектор (обязательно конвертируем в list для JSON)
        query_vector = self.user_vectors[u_idx].tolist()
        
        # 3. Формируем запрос через модели (это работает в любой версии)
        search_request = rest_models.SearchRequest(
            vector=query_vector,
            limit=k,
            with_payload=False, # Нам нужны только ID для ранкера
            score_threshold=0.0
        )
        
        # try:
        #     # 4. Выполняем запрос через Points API
        #     api_result = self.client.http.points_api.search_points(
        #         collection_name=self.collection_name,
        #         search_points=search_request
        #     )
            
        #     # В разных версиях ответ может быть обернут по-разному
        #     # Обычно это объект, у которого есть атрибут result
        #     hits = api_result.result if hasattr(api_result, 'result') else api_result
            
        #     # Извлекаем ID
        #     candidate_ids = [hit.id for hit in hits]
        #     return candidate_ids
            
        # except Exception as e:
        #     print(f"❌ Qdrant Search Error: {e}")
            # return []

        try:
            # ИСПОЛЬЗУЕМ ОБЫЧНЫЙ ВЫСОКОУРОВНЕВЫЙ API
            # В Docker у нас свежая либа, это будет работать.
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=k,
                with_payload=False
            )
            
            # Извлекаем ID
            candidate_ids = [hit.id for hit in search_result]
            return candidate_ids
            
        except Exception as e:
            # Логируем реальную ошибку, чтобы видеть детали
            print(f"❌ Qdrant Search Error: {e}")
            return []

    def search_by_vector(self, vector: List[float], k=10) -> List[dict]:
        """
        Хелпер для отладки: поиск с возвратом названий
        """
        search_request = rest_models.SearchRequest(
            vector=vector,
            limit=k,
            with_payload=True
        )
        
        try:
            api_result = self.client.http.points_api.search_points(
                collection_name=self.collection_name,
                search_points=search_request
            )
            hits = api_result.result if hasattr(api_result, 'result') else api_result
            
            # Извлекаем данные для UI
            results = []
            for hit in hits:
                payload = hit.payload
                # Payload может быть объектом или dict в зависимости от версии
                title = payload.get('title') if isinstance(payload, dict) else getattr(payload, 'title', 'Unknown')
                
                results.append({
                    "id": hit.id,
                    "title": title,
                    "score": hit.score
                })
            return results
            
        except Exception as e:
            print(f"❌ Qdrant Debug Search Error: {e}")
            return []