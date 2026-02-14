from locust import HttpUser, task, between, constant
import random

class RecSysUser(HttpUser):
    # ВАЖНО: 
    # between(1, 3) — это имитация ПОВЕДЕНИЯ пользователя (он читает, думает).
    # Это хорошо для проверки "выдержим ли мы 1000 юзеров на сайте".
    # Но если вы ищете ПРЕДЕЛ производительности (Max RPS), лучше убрать паузы 
    # или поставить constant(0), чтобы долбить сервер без остановки.
    # Для начала оставьте between, это реалистичнее.
    wait_time = between(0.5, 2) 

    @task
    def get_recommendations(self):
        user_id = random.randint(1, 6040)
        
        payload = {
            "user_id": user_id,
            "top_k": 10
        }
        
        # catch_response=True нужен, чтобы самим решать, что считать ошибкой
        with self.client.post("/recommend", json=payload, catch_response=True) as response:
            
            # 1. Сначала проверяем технические ошибки (500, 502, 404)
            if response.status_code != 200:
                response.failure(f"Status code: {response.status_code} | Text: {response.text}")
                return

            response.success()