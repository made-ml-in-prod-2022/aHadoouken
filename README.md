ML project
==============================

Выполненная Домашняя работа №1 по курсу "ML в проде"

Датасет для обучения модели: https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci

Использование проекта
==============================
**Установка:**
~~~
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
~~~

**Обучение модели:**
~~~
python ml_app/main.py --mode=train --config=configs/config_lr.yaml
~~~
где

--config - путь к конфиг файлу с параметрами обучения

**Использование модели:**
~~~
python ml_app/main.py --mode=predict --config=configs/config_pred.yaml
~~~
где

--config - путь к конфиг файлу с параметрами предсказания

**Тестирование модели:**
~~~
python ml_app/tests/test.py
~~~
Онлайн использование (HW2)
==============================
Перед запуском сервера следует также установить все зависимости:
~~~
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
~~~
Далее необходимо задать переменные среды для путей моделей:
~~~
export PATH_TO_MODEL_LR=online_inference/models/model_lr.pkl
export PATH_TO_MODEL_RF=online_inference/models/model_rf.pkl
~~~
Сервис написан с помощью библиотеки FastAPI, запускается командой:
~~~
uvicorn --app-dir=online_inference app:app --host 0.0.0.0 --port 8000
~~~
Скрипт отправки реквестов запускается с помощью команды:
~~~
python online_inference/script.py --data_path=online_inference/data/data_s.csv [FLAGS]
~~~
С использованием флагов:

--data_path - путь к файлу с данными, обязательный флаг

--model - тип модели для предикта (lr/rf) (default-lr)

--host - ip адресс сервиса (default - 0.0.0.0)

--port - порт (default - 8000)

Тесты:
~~~
pytest online_inference/test_app.py
~~~

**Docker:**

Запуск сервиса с помощью докера осуществляется с помощью команд:
~~~
docker build -t model:v1 .
docker run -p 8000:8000 model:v1
~~~
Либо при помощи DockerHub:
~~~
docker pull ahadoouken/model:v1
docker run -p 8000:8000 ahadoouken/model:v1
~~~

Архитектура проекта
==============================

    ├── LICENSE
    ├── Makefile           
    ├── README.md          
    ├── data
    │   ├── results        <- Папка, содержащая результаты модели
    │   └── raw            <- Папка для сырых данных для обучения или предсказания
    │
    ├── notebooks          <- Содержит EDA и отчет по данным
    │
    ├── configs            <- Готовые конфиги с параметрами обучения модели
    │
    ├── lints              <- Содержит скрипт запуска pylint
    │
    ├── requirements.txt   <- Файл с зависимостями
    │
    ├── ml_app             <- Исходный код проекта
    │   ├── __init__.py    
    │   │
    │   ├── data           <- Скрипт для загрузки данных
    │   │   └── obtain_data.py
    │   │
    │   ├── features       <- Содержит класс кастомного трансформера
    │   │   └── preproc_data.py
    │   │
    │   ├── models         <- Скрипт для работы с моделью
    │   │   └── model_utils.py
    │   │
    │   ├── utils          <- Модуль содержащий DataClass для считываения параметров
    │   │   ├── data_params.py
    │   │   ├── model_params.py
    │   │   └── preproc_params.py
    │   │
    │   └── tests          <- Юнит тесты
    │
    │
    │
    └── setup.py            


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
