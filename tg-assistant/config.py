from dotenv import load_dotenv
import os

# Загрузка переменных окружения из файла .env``
load_dotenv()

# Хост и порт сервера API
HOST = os.getenv("API_HOST", "http://127.0.0.1")
PORT = os.getenv("API_PORT", "8000")

# URL для получения ответа от сервера
ANSWER_URL = f"https://{HOST}:{PORT}/api/v1/chat/answer"
FEEDBACK_URL = f"https://{HOST}:{PORT}/api/v1/analytic/"

# Токен бота
TG_TOKEN = os.getenv("TG_TOKEN")
