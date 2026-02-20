import random
from locust import HttpUser, task, between, events

# --- Настройки теста ---
TOTAL_USERS = 1000  # Твои игрушечные данные
SLA_REDIS_MS = 50   # Если Redis дольше 50мс - это провал
SLA_TOTAL_MS = 200  # Если всё вместе дольше 200мс - это провал
SLA_QDRANT_MS = 30 # Если Qdrant дольше 100мс - это провал

class RecSysUser(HttpUser):
    # wait_time = between(0.5, 2)  # Имитация реального человека (для проверки стабильности)
    wait_time = between(0.1, 0.5) # Агрессивный тест (для поиска Max RPS)

    @task
    def get_recommendations(self):
        # 1. Генерируем случайного пользователя, чтобы пробить кэш
        user_id = random.randint(1, TOTAL_USERS)
        
        payload = {
            "user_id": user_id,
            "top_k": 10
        }

        # 2. Делаем запрос
        # catch_response=True позволяет нам самим решать, что считать ошибкой (fail)
        with self.client.post("/recommend", json=payload, catch_response=True) as response:
            
            # --- Сценарий A: Сервер ответил 200 OK ---
            if response.status_code == 200:
                try:
                    # Извлекаем тайминги из заголовков (которые мы добавили в FastAPI)
                    # Если заголовка нет, считаем 0, чтобы код не упал
                    redis_time = float(response.headers.get("X-Latency-Features", 0))
                    qdrant_time = float(response.headers.get("X-Latency-Retrieval", 0))
                    model_time = float(response.headers.get("X-Latency-Ranking", 0))
                    total_time = response.elapsed.total_seconds() * 1000

                    # --- ВАЖНО: Отправляем эти метрики в Locust как отдельные события ---
                    # Это создаст в таблице статистики строки "Internal_Redis", "Internal_Qdrant"
                    # Ты увидишь их среднее время и P95 отдельно от основного запроса!
                    self.environment.events.request.fire(
                        request_type="INTERNAL",
                        name="Step_Redis",
                        response_time=redis_time,
                        response_length=0,
                        exception=None,
                    )
                    self.environment.events.request.fire(
                        request_type="INTERNAL",
                        name="Step_Qdrant",
                        response_time=qdrant_time,
                        response_length=0,
                        exception=None,
                    )
                    self.environment.events.request.fire(
                        request_type="INTERNAL",
                        name="Step_CatBoost",
                        response_time=model_time,
                        response_length=0,
                        exception=None,
                    )

                    # --- Проверка SLA (Почему мы считаем запрос проваленным?) ---
                    if redis_time > SLA_REDIS_MS:
                        response.failure(f"SLA Violated: Redis slow ({redis_time:.1f}ms)")
                    
                    elif total_time > SLA_TOTAL_MS:
                        response.failure(f"SLA Violated: Total slow ({total_time:.1f}ms)")

                    elif qdrant_time > SLA_QDRANT_MS:  
                        response.failure(f"SLA Violated: Qdrant slow ({qdrant_time:.1f}ms)")
                    
                    else:
                        # Всё отлично
                        response.success()

                except ValueError:
                    # Если заголовки пришли битые
                    response.failure("Response headers format error")

            # --- Сценарий B: Сервер упал с ошибкой ---
            elif response.status_code == 503:
                # Наша кастомная ошибка "Redis умер"
                response.failure("CRITICAL: Feature Store Unavailable (503)")
            
            elif response.status_code == 500:
                response.failure("CRITICAL: Server Error (500)")
            
            else:
                response.failure(f"HTTP Error {response.status_code}")