# 🎬 Movie Success & Budget Prediction

Проект по анализу и прогнозированию успешности фильмов на основе данных TMDB: предсказание успеха фильма (классификация), предсказание бюджета (регрессия), кластеризация фильмов и NLP-анализ описаний (overview) для выявления слов, связанных с высоким рейтингом.

Репозиторий: [github.com/RONEW2J/movie_success-budget_prediction](https://github.com/RONEW2J/movie_success-budget_prediction)

## 📊 Датасет

Проект использует датасет **TMDB Movies Daily Updates** с Kaggle:

🔗 [kaggle.com/datasets/alanvourch/tmdb-movies-daily-updates](https://www.kaggle.com/datasets/alanvourch/tmdb-movies-daily-updates)

Датасет содержит подробную информацию о сотнях тысяч фильмов из The Movie Database (TMDB), включая:
- бюджет (`budget`) и сборы (`revenue`)
- рейтинг и количество голосов (`vote_average`, `vote_count`)
- жанры, дату выхода, продолжительность (`runtime`)
- описание сюжета (`overview`), теглайн
- актёров, режиссёров, сценаристов, продюсеров, композиторов (`cast`, `director`, `writers`, `producers`, `music_composer`)
- постеры (`poster_path`) и другую метаинформацию

Файл нужно скачать с Kaggle и положить в проект как `data/TMDB_all_movies.csv`.

## 🧠 Что делает проект

На основе данных вычисляются производные признаки:
- **profit** = revenue − budget
- **roi** = (revenue − budget) / budget
- **success** = revenue > budget (бинарный признак успеха)
- **primary_genre** — основной жанр фильма
- признаки даты выхода (год, месяц)

Реализованы несколько ML-направлений:

1. **Предсказание успеха фильма** — классификация (Gradient Boosting) на основе бюджета, жанра, актёрского состава, времени выхода и др.
2. **Предсказание бюджета** — регрессия (Random Forest) для оценки нужного бюджета по параметрам фильма.
3. **Кластеризация фильмов** — K-Means по числовым признакам (бюджет, рейтинг, ROI и т.д.) для сегментации фильмов на группы со схожими характеристиками.
4. **NLP-анализ описаний (overview)** — токенизация текста, удаление стоп-слов, выявление ключевых слов и жанровых тем, а также сравнение частотности слов в «успешных» (рейтинг > 6.5) и «неуспешных» фильмах.
5. **Интерактивный дашборд на Streamlit** — визуализация трендов индустрии, анализ жанров, бюджет/сборы, обучение моделей и построение прогнозов прямо из интерфейса.

## 📁 Структура репозитория

| Файл | Описание |
|---|---|
| `streamlit_ml_dashboard.py` | Основное Streamlit-приложение: EDA, NLP-анализ, обучение и инференс моделей (success prediction, budget prediction, K-Means) |
| `project_ml.ipynb` | Основной ноутбук с ML-пайплайном (обучение моделей) |
| `budget_prediction.ipynb` | Ноутбук с моделью предсказания бюджета фильма (Random Forest Regressor) |
| `k-mean_movie_datatset.ipynb` | Ноутбук с кластеризацией фильмов методом K-Means |
| `nlp_overview_of_movies.ipynb` | Ноутбук с NLP-анализом описаний фильмов |
| `top_profit_movies.html` | Экспортированная визуализация топ фильмов по прибыли |
| `top_rated_movies.html` | Экспортированная визуализация топ фильмов по рейтингу |
| `top_roi_movies.html` | Экспортированная визуализация топ фильмов по ROI |
| `requirements.txt` | Список зависимостей проекта |

## 🛠 Стек технологий

- **Данные и анализ:** pandas, numpy
- **ML:** scikit-learn (RandomForestRegressor, GradientBoostingClassifier, KMeans, StandardScaler, TF-IDF)
- **NLP:** собственная токенизация + удаление стоп-слов, ключевые слова по жанрам, простой rule-based сентимент-анализ
- **Визуализация:** plotly, matplotlib, seaborn, altair
- **Дашборд:** Streamlit
- **Окружение:** Jupyter Notebook / JupyterLab

## 🚀 Установка и запуск

### 1. Клонировать репозиторий

```bash
git clone https://github.com/RONEW2J/movie_success-budget_prediction.git
cd movie_success-budget_prediction
```

### 2. Создать виртуальное окружение и установить зависимости

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Скачать датасет

Скачайте `TMDB_all_movies.csv` с [Kaggle](https://www.kaggle.com/datasets/alanvourch/tmdb-movies-daily-updates) и поместите его в:

```
data/TMDB_all_movies.csv
```

### 4. Обучить модели (через ноутбуки)

Запустите по очереди в Jupyter:

```bash
jupyter lab
```

- `project_ml.ipynb` — общий пайплайн предобработки и обучения
- `budget_prediction.ipynb` — модель предсказания бюджета
- `k-mean_movie_datatset.ipynb` — кластеризация
- `nlp_overview_of_movies.ipynb` — NLP-анализ

Обученные модели сохраняются в папку `models/` в виде `.pkl` файлов (модель, scaler, метаданные) — они используются дашбордом.

### 5. Запустить дашборд

```bash
streamlit run streamlit_ml_dashboard.py
```

После запуска дашборд будет доступен в браузере (обычно `http://localhost:8501`).

## 📈 Возможности дашборда

- **Overview** — общая статистика по датасету и статус загруженных моделей
- **Data Exploration** — тренды индустрии по годам, анализ жанров, соотношение бюджет/сборы (с постерами и интерактивными графиками)
- **NLP Overview Analysis** — анализ слов в описаниях успешных/неуспешных фильмов, тестирование собственного описания
- **Success Prediction** — прогноз вероятности успеха фильма по параметрам (бюджет, жанр, актёры, режиссёр и т.д.)
- **Budget Prediction** — оценка необходимого бюджета
- **K-Means Clustering** — визуализация кластеров фильмов
- **Model Comparison** — сравнение метрик обученных моделей

## 📌 Примечания

- Модели тренируются на фильмах со статусом `Released`, положительным бюджетом/сборами и минимум 10 голосами.
- Порог "успешного" описания в NLP-модуле — рейтинг **> 6.5**.
- "Прибыльным" в разделе финансового анализа считается фильм, чьи сборы превышают бюджет **в 2 и более раз** (ROI ≥ 100%).

## 📄 Лицензия

Лицензия не указана в репозитории. Датасет TMDB используется в соответствии с условиями Kaggle и TMDB API.
