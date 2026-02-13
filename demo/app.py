import gradio as gr
import pandas as pd
from typing import Dict, Any

# Импортируем наши слои
from src.inference import ModelService      # Твой класс ранжирования
from src.retrieval import AnnoyRetriever    # Поиск кандидатов (Annoy)
from src.stores import InMemoryFeatureStore # Хранилище юзеров

# --- 1. Инициализация (Singleton) ---
print("🚀 Starting Demo App with Service Layer...")

# A. Сервис ранжирования (внутри него модель и фичи айтемов)
model_service = ModelService()
model_service.load()

# B. Поиск кандидатов (Annoy index)
retriever = AnnoyRetriever()

# C. Хранилище фичей юзеров (нам нужно достать Gender/Age перед отправкой в сервис)
# Используем тот же класс, что писали ранее, но будем брать только юзеров
user_store = InMemoryFeatureStore() 

# --- 2. Оркестрация (Controller Logic) ---

def get_recommendations(user_id_str: str, top_k: int = 20):
    """
    Эта функция играет роль 'Controller' в MVC.
    Она вызывает разные сервисы и собирает ответ.
    """
    try:
        user_id = int(user_id_str)
    except ValueError:
        return "❌ Error: Invalid User ID", pd.DataFrame()

    # 1. Candidate Generation (Retrieval Layer)
    # ----------------------------------------
    # Ищем 100 похожих фильмов по вектору
    candidate_ids = retriever.get_candidates(user_id, k=100)
    
    if not candidate_ids:
        # Fallback: Если для юзера нет эмбеддинга (Cold Start),
        # берем топ популярных из сервиса
        candidate_ids = model_service.get_top_popular(100)
        retrieval_info = "⚠️ Cold User: Using Popular Items fallback"
    else:
        retrieval_info = f"✅ Annoy Retrieval: Found {len(candidate_ids)} candidates based on embeddings."

    # 2. Fetch User Context (Feature Store Layer)
    # ------------------------------------------
    # Нам нужны фичи юзера (Age, Gender), чтобы подать их в CatBoost
    user_features = user_store.get_user_features(user_id)
    
    # 3. Re-Ranking (Inference Layer)
    # ------------------------------
    # Вызываем твой сервис! Он сам заджойнит айтемы и прогонит модель.
    ranked_items = model_service.predict(user_features, candidate_ids)
    
    # 4. Presentation Layer
    # ---------------------
    # Превращаем список словарей в DataFrame для таблицы
    if not ranked_items:
        return retrieval_info, pd.DataFrame()

    df = pd.DataFrame(ranked_items)
    
    # Оставляем красивые колонки
    df = df[["movie_id", "title", "genres", "score"]]
    df.columns = ["ID", "Title", "Genres", "Score"]
    
    return retrieval_info, df.head(top_k)


# --- 3. Интерфейс (View) ---

with gr.Blocks(title="RecSys Production Demo") as demo:
    gr.Markdown("# 🎬 RecSys: The Full Pipeline")
    gr.Markdown("Architecture: **ALS (Annoy)** -> **Feature Store** -> **CatBoost Ranker (Service)**")
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 👤 User Context")
            user_input = gr.Textbox(label="User ID", value="1", placeholder="Enter ID (1-6040)")
            
            # Показываем, кто этот юзер
            user_meta_json = gr.JSON(label="User Features")
            
            btn = gr.Button("🚀 Recommend", variant="primary")
            
        with gr.Column(scale=2):
            status_box = gr.Textbox(label="Pipeline Status", interactive=False)
            output_table = gr.Dataframe(
                headers=["ID", "Title", "Genres", "Score"],
                datatype=["number", "str", "str", "number"],
                label="Final Ranking"
            )

    # Интерактив: показать фичи юзера при вводе ID
    def show_user_details(uid):
        try:
            return user_store.get_user_features(int(uid))
        except:
            return {"error": "User not found"}
            
    user_input.change(fn=show_user_details, inputs=user_input, outputs=user_meta_json)
    print(user_input)
    btn.click(
        fn=get_recommendations,
        inputs=[user_input],
        outputs=[status_box, output_table]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)