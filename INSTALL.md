### 1. Клонирование репозитория

Сначала необходимо склонировать проект на локальную машину или сервер:

     Bash/PowerShell
```
git clone <ссылка_на_ваш_репозиторий>
cd <название_папки_проекта>
```

---

### 2. Подготовка окружения

#### **Для Windows (PowerShell / CMD)**

- **Создать виртуальное окружение:**

    
    PowerShell
    
    ```
    python -m venv venv
    ```


- **Активировать окружение:**

    
    PowerShell
    
    ```
    .\venv\Scripts\activate
    ```
    
- **Установить зависимости:**
    
    PowerShell
    
    ```
    python -m pip install --upgrade pip
    pip install -r requirements.txt
    ```
    

#### **Для Linux

- **Установить системные библиотеки** (необходимы для работы OpenCV):

    
    Bash
    
    ```
    sudo apt-get update && sudo apt-get install -y libgl1 libglib2.0-0
    ```
    
- **Создать виртуальное окружение:**
    
    Bash
    
    ```
    python3 -m venv venv
    ```
    
- **Активировать окружение:**
    
    Bash
    
    ```
    source venv/bin/activate
    ```
    
- **Установить зависимости:**
    
    Bash
    
    ```
    pip install --upgrade pip
    pip install -r requirements.txt
    ```
    

---

### 3. Запуск приложения

Для старта интерфейса Streamlit использовать соответствующую команду:

```
    streamlit run app.py
```   

---

### 4. Завершение запуска и доступ к интерфейсу

**Что отобразится в командной строке при успешном запуске:** После выполнения команды терминал выведет информационное сообщение:

```
  You can now view your Streamlit app in your browser.

  Local URL: http://localhost:8501
  Network URL: http://192.168.1.100:8501
```

**Доступ к приложению:**

Приложение откроется в браузере по умолчанию автоматически. 
Если этого не произошло, нужно перейти в браузере по адресу `http://localhost:8501`. 