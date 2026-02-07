# Используем стабильный Python
FROM python:3.9-slim

# Устанавливаем системные зависимости для OpenCV и графики
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Создаем рабочую директорию
WORKDIR /app

# Сначала копируем только requirements, чтобы закэшировать установку библиотек
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем остальной код
COPY . .

# Открываем порт, который использует Streamlit по умолчанию
EXPOSE 8501

# Команда для запуска (с отключением проверки обновлений и настройки порта)
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]