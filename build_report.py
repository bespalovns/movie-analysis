from pathlib import Path
import base64
import html
import json
import unicodedata

import numpy as np
import pandas as pd

from bokeh.embed import components
from bokeh.models import ColumnDataSource, HoverTool
from bokeh.plotting import figure
from bokeh.resources import INLINE


# ============================================================
# 1. Настройки
# ============================================================

AUTHOR = "Беспалов Николай Сергеевич"
GROUP = "ИСТZ-3401"

BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
OUTPUT_DIR = BASE_DIR / "output"
PHOTO_PATH = BASE_DIR / "assets" / "photo.jpg"

SOURCE_URL = (
    "https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata"
)

# Оценки фильмов с меньшим числом голосов не используем
# в графиках рейтингов: такие оценки менее устойчивы.
MIN_VOTES = 50

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GREEN = "#FF6600"
YELLOW = "#FFF0E5"
TOOLS = "pan,wheel_zoom,box_zoom,reset,save"


# ============================================================
# 2. Загрузка и первоначальная проверка
# ============================================================

movies = pd.read_csv(
    RAW_DIR / "tmdb_5000_movies.csv",
    encoding="utf-8",
)

credits = pd.read_csv(
    RAW_DIR / "tmdb_5000_credits.csv",
    encoding="utf-8",
)

quality = {
    "Исходное число строк movies": len(movies),
    "Исходное число строк credits": len(credits),
    "Полные дубликаты movies": int(movies.duplicated().sum()),
    "Полные дубликаты credits": int(credits.duplicated().sum()),
}

initial_missing = {
    column: int(count)
    for column, count in movies.isna().sum().items()
}

# Удаляем только полные дубликаты и полностью пустые строки.
movies = movies.dropna(how="all").drop_duplicates().copy()
credits = credits.dropna(how="all").drop_duplicates().copy()


def normalize_text(value):
    """Приведение текста к единому представлению."""
    if not isinstance(value, str):
        return value

    value = unicodedata.normalize("NFC", value)
    value = value.replace("\u00a0", " ").strip()

    return value if value else np.nan


for frame in (movies, credits):
    for column in frame.select_dtypes(
        include=["object", "string"]
    ).columns:
        frame[column] = frame[column].map(normalize_text)


# ============================================================
# 3. Проверка идентификаторов и объединение
# ============================================================

movies["id"] = pd.to_numeric(movies["id"], errors="raise")
credits["movie_id"] = pd.to_numeric(
    credits["movie_id"], errors="raise"
)

if movies["id"].isna().any() or credits["movie_id"].isna().any():
    raise ValueError("Обнаружены пустые идентификаторы фильмов.")

if movies["id"].duplicated().any():
    raise ValueError(
        "В movies есть повторяющиеся ID. "
        "Нужно проверить конфликтующие записи."
    )

if credits["movie_id"].duplicated().any():
    raise ValueError(
        "В credits есть повторяющиеся ID. "
        "Нужно проверить конфликтующие записи."
    )

quality["Фильмы без соответствия в credits"] = int(
    (~movies["id"].isin(credits["movie_id"])).sum()
)

quality["Записи credits без соответствия в movies"] = int(
    (~credits["movie_id"].isin(movies["id"])).sum()
)

df = movies.merge(
    credits[["movie_id", "cast", "crew"]],
    left_on="id",
    right_on="movie_id",
    how="left",
    validate="one_to_one",
)

df = df.drop(columns="movie_id")


# ============================================================
# 4. Обработка чисел и дат
# ============================================================

numeric_columns = [
    "budget",
    "revenue",
    "runtime",
    "popularity",
    "vote_average",
    "vote_count",
]

for column in numeric_columns:
    original = df[column]
    converted = pd.to_numeric(original, errors="coerce")

    invalid = original.notna() & converted.isna()
    quality[f"Некорректные числа: {column}"] = int(invalid.sum())

    nonfinite = converted.notna() & ~np.isfinite(converted)
    quality[f"Бесконечные значения: {column}"] = int(
        nonfinite.sum()
    )

    converted = converted.mask(nonfinite)
    negative = converted < 0

    quality[f"Отрицательные значения: {column}"] = int(
        negative.sum()
    )

    df[column] = converted.mask(negative)


dates = pd.to_datetime(df["release_date"], errors="coerce")

quality["Пустые или некорректные даты выхода"] = int(
    dates.isna().sum()
)

df["release_date"] = dates
df["release_year"] = dates.dt.year.astype("Int64")

# Исходные числовые столбцы сохраняем.
# Для анализа создаём отдельные столбцы.
for original, cleaned in [
    ("budget", "budget_clean"),
    ("revenue", "revenue_clean"),
    ("runtime", "runtime_clean"),
]:
    quality[f"Нулевые значения: {original}"] = int(
        df[original].eq(0).sum()
    )
    df[cleaned] = df[original].where(df[original] > 0)

# Рейтинг имеет смысл при наличии хотя бы одного голоса.
valid_rating = (
    df["vote_average"].between(0, 10)
    & df["vote_count"].gt(0)
)

df["rating_clean"] = df["vote_average"].where(valid_rating)

quality["Фильмы без пригодной для анализа оценки"] = int(
    df["rating_clean"].isna().sum()
)


# ============================================================
# 5. Разбор JSON и извлечение режиссёров
# ============================================================

def parse_json_list(value):
    if pd.isna(value):
        return []

    result = json.loads(value)

    if not isinstance(result, list):
        raise ValueError("В JSON-поле ожидался список.")

    return result


df["genre_list"] = df["genres"].map(
    lambda value: [
        item["name"]
        for item in parse_json_list(value)
        if item.get("name")
    ]
)

df["director_list"] = df["crew"].map(
    lambda value: [
        item["name"]
        for item in parse_json_list(value)
        if item.get("job") == "Director" and item.get("name")
    ]
)

df["cast_count"] = df["cast"].map(
    lambda value: len(parse_json_list(value))
)

quality["Фильмы без указанного жанра"] = int(
    df["genre_list"].map(len).eq(0).sum()
)

quality["Фильмы без указанного режиссёра"] = int(
    df["director_list"].map(len).eq(0).sum()
)

quality["Итоговое число фильмов"] = len(df)

# Один фильм может иметь несколько жанров.
# Здесь каждая строка представляет связь «фильм - жанр».
genre_data = (
    df[["id", "title", "genre_list", "rating_clean", "vote_count"]]
    .explode("genre_list")
    .dropna(subset=["genre_list"])
    .drop_duplicates(subset=["id", "genre_list"])
    .rename(columns={"genre_list": "genre"})
)

director_data = (
    df[["id", "director_list"]]
    .explode("director_list")
    .dropna(subset=["director_list"])
    .drop_duplicates(subset=["id", "director_list"])
    .rename(columns={"director_list": "director"})
)


# ============================================================
# 6. Нормализация Min-Max
# ============================================================

normalization = {}

for column in [
    "budget_clean",
    "revenue_clean",
    "runtime_clean",
    "rating_clean",
    "popularity",
]:
    minimum = df[column].min()
    maximum = df[column].max()

    target = column + "_norm"

    if pd.isna(minimum) or pd.isna(maximum):
        df[target] = np.nan
    elif maximum == minimum:
        df[target] = df[column].where(
            df[column].isna(), 0.0
        )
    else:
        df[target] = (
            (df[column] - minimum) / (maximum - minimum)
        )

    normalization[column] = {
        "minimum": None if pd.isna(minimum) else float(minimum),
        "maximum": None if pd.isna(maximum) else float(maximum),
        "method": "Min-Max",
    }


# ============================================================
# 7. Сохранение подготовленных данных
# ============================================================

export_columns = [
    "id",
    "title",
    "original_language",
    "release_date",
    "release_year",
    "budget",
    "budget_clean",
    "revenue",
    "revenue_clean",
    "runtime",
    "runtime_clean",
    "vote_average",
    "rating_clean",
    "vote_count",
    "popularity",
    "cast_count",
    "genre_list",
    "director_list",
] + [
    column for column in df.columns if column.endswith("_norm")
]

export_df = df[export_columns].copy()

for column in ["genre_list", "director_list"]:
    export_df[column] = export_df[column].map(
        lambda value: json.dumps(value, ensure_ascii=False)
    )

export_df.to_csv(
    OUTPUT_DIR / "movies_clean.csv",
    index=False,
    encoding="utf-8-sig",
)

genre_data.to_csv(
    OUTPUT_DIR / "movie_genres.csv",
    index=False,
    encoding="utf-8-sig",
)

audit = {
    "checks": quality,
    "initial_missing_movies": initial_missing,
    "normalization": normalization,
}

(OUTPUT_DIR / "quality_report.json").write_text(
    json.dumps(audit, ensure_ascii=False, indent=2),
    encoding="utf-8",
)


# ============================================================
# 8. Общие функции построения графиков
# ============================================================

charts = []


def register(plot, group, explanation):
    plot.background_fill_color = "#FFFFFF"
    plot.border_fill_color = "#FFFFFF"
    plot.outline_line_color = "#E6E6E6"

    plot.title.text_font_size = "15pt"
    plot.title.text_color = "#111111"

    plot.axis.axis_label_text_color = "#111111"
    plot.axis.major_label_text_color = "#333333"
    plot.axis.axis_line_color = "#CCCCCC"
    plot.axis.major_tick_line_color = "#CCCCCC"
    plot.axis.minor_tick_line_color = None

    plot.grid.grid_line_color = "#EEEEEE"

    charts.append({
        "plot": plot,
        "group": group,
        "explanation": explanation,
    })


def bar_chart(series, title, value_label, group, explanation):
    series = series.dropna()
    labels = [str(value) for value in series.index]

    source = ColumnDataSource({
        "label": labels,
        "value": series.to_numpy(),
    })

    plot = figure(
        title=title,
        y_range=labels[::-1],
        height=max(340, len(labels) * 27),
        sizing_mode="stretch_width",
        tools=TOOLS,
        x_axis_label=value_label,
    )

    plot.hbar(
        y="label",
        right="value",
        height=0.7,
        source=source,
        color=GREEN,
    )

    plot.x_range.start = 0

    plot.add_tools(HoverTool(tooltips=[
        ("Категория", "@label"),
        ("Значение", "@value{0,0.00}"),
    ]))

    register(plot, group, explanation)


def histogram(values, title, x_label, group, explanation):
    values = values.dropna()

    counts, edges = np.histogram(values, bins=25)

    source = ColumnDataSource({
        "left": edges[:-1],
        "right": edges[1:],
        "count": counts,
    })

    plot = figure(
        title=title,
        height=370,
        sizing_mode="stretch_width",
        tools=TOOLS,
        x_axis_label=x_label,
        y_axis_label="Количество фильмов",
    )

    plot.quad(
        top="count",
        bottom=0,
        left="left",
        right="right",
        source=source,
        fill_color=GREEN,
        line_color="white",
    )

    plot.y_range.start = 0

    plot.add_tools(HoverTool(tooltips=[
        ("От", "@left{0.00}"),
        ("До", "@right{0.00}"),
        ("Фильмов", "@count"),
    ]))

    register(plot, group, explanation)


def scatter_chart(
    frame, x, y, title, x_label, y_label,
    group, explanation, x_log=False, y_log=False
):
    data = frame[["title", x, y]].dropna().copy()

    if x_log:
        data = data[data[x] > 0]

    if y_log:
        data = data[data[y] > 0]

    source = ColumnDataSource({
        "title": data["title"].fillna("Без названия"),
        "x": data[x],
        "y": data[y],
    })

    plot = figure(
        title=title,
        height=420,
        sizing_mode="stretch_width",
        tools=TOOLS,
        x_axis_label=x_label,
        y_axis_label=y_label,
        x_axis_type="log" if x_log else "linear",
        y_axis_type="log" if y_log else "linear",
    )

    plot.scatter(
        x="x",
        y="y",
        source=source,
        size=6,
        alpha=0.45,
        color=GREEN,
    )

    plot.add_tools(HoverTool(tooltips=[
        ("Фильм", "@title"),
        (x_label, "@x{0,0.00}"),
        (y_label, "@y{0,0.00}"),
    ]))

    register(
        plot,
        group,
        explanation + f" Использовано фильмов: {len(data)}.",
    )


# ============================================================
# 9. Подготовка выборок
# ============================================================

rated = df[
    df["rating_clean"].notna()
    & df["vote_count"].ge(MIN_VOTES)
].copy()

financial = df.dropna(
    subset=["budget_clean", "revenue_clean"]
).copy()

genre_counts = (
    genre_data.groupby("genre")["id"]
    .nunique()
    .sort_values(ascending=False)
)

director_counts = (
    director_data.groupby("director")["id"]
    .nunique()
    .sort_values(ascending=False)
)

year_counts = (
    df.dropna(subset=["release_year"])
    .groupby("release_year")
    .size()
    .sort_index()
)

genre_rating = (
    genre_data[
        genre_data["vote_count"].ge(MIN_VOTES)
        & genre_data["rating_clean"].notna()
    ]
    .groupby("genre")["rating_clean"]
    .agg(["median", "count"])
)

# Не сравниваем категории, представленные единичными фильмами.
genre_rating = (
    genre_rating[genre_rating["count"] >= 30]
    .sort_values("median", ascending=False)
)

quality["Фильмы в финансовом анализе"] = len(financial)
quality["Фильмы в анализе рейтингов"] = len(rated)

# Обновляем отчёт после формирования аналитических выборок.
audit["checks"] = quality

(OUTPUT_DIR / "quality_report.json").write_text(
    json.dumps(audit, ensure_ascii=False, indent=2),
    encoding="utf-8",
)


# ============================================================
# 10. Десять визуализаций
# ============================================================

plot = figure(
    title="1. Распределение фильмов выборки по годам",
    height=370,
    sizing_mode="stretch_width",
    tools=TOOLS,
    x_axis_label="Год выхода",
    y_axis_label="Количество фильмов",
)

year_source = ColumnDataSource({
    "year": year_counts.index.to_numpy(dtype=int),
    "count": year_counts.to_numpy(),
})

plot.line(
    "year", "count", source=year_source,
    line_width=3, color=GREEN
)

plot.scatter(
    "year", "count", source=year_source,
    size=5, color=GREEN
)

plot.add_tools(HoverTool(tooltips=[
    ("Год", "@year"),
    ("Фильмов", "@count"),
]))

register(
    plot,
    "Состав выборки",
    "Показано число фильмов каждого года именно в датасете. "
    "Это не статистика всего мирового кинопроизводства. "
    "Записи без даты исключены только из этого графика.",
)

bar_chart(
    genre_counts,
    "2. Представленность жанров",
    "Количество фильмов",
    "Состав выборки",
    "Фильм учитывается в каждом указанном у него жанре. "
    "Поэтому сумма столбцов может превышать число фильмов.",
)

bar_chart(
    df["original_language"].value_counts().head(10),
    "3. Десять наиболее частых исходных языков",
    "Количество фильмов",
    "Состав выборки",
    "Показан исходный язык фильма. "
    "Он не равнозначен стране производства.",
)

bar_chart(
    director_counts.head(10),
    "4. Режиссёры по числу фильмов в выборке",
    "Количество уникальных фильмов",
    "Состав выборки",
    "Показаны десять режиссёров с наибольшим числом фильмов "
    "в датасете. При совместной режиссуре фильм учитывается "
    "у каждого режиссёра. Это не рейтинг качества работ.",
)

histogram(
    rated["rating_clean"],
    "5. Распределение зрительских рейтингов",
    "Оценка от 0 до 10",
    "Рейтинги и продолжительность",
    f"Включены фильмы минимум с {MIN_VOTES} голосами. "
    "Порог уменьшает влияние оценок с очень малым числом голосов, "
    "но не устраняет все возможные смещения.",
)

histogram(
    df["runtime_clean"],
    "6. Распределение продолжительности",
    "Продолжительность, минут",
    "Рейтинги и продолжительность",
    "Нулевая, отрицательная и неизвестная продолжительность "
    "не участвует в расчёте. Большие положительные значения "
    "не удаляются автоматически как выбросы.",
)

bar_chart(
    genre_rating["median"],
    "7. Медианный рейтинг по жанрам",
    "Медианная оценка",
    "Рейтинги и продолжительность",
    f"Использованы фильмы минимум с {MIN_VOTES} голосами. "
    "Показаны жанры минимум с 30 пригодными оценками. "
    "Различия медиан сами по себе не доказывают "
    "статистическую значимость или влияние жанра.",
)

scatter_chart(
    financial,
    "budget_clean",
    "revenue_clean",
    "8. Бюджеты и кассовые сборы",
    "Бюджет, долларов США",
    "Сборы, долларов США",
    "Бюджеты и кассовые сборы",
    "Каждая точка - фильм с положительными бюджетом и сборами. "
    "Обе оси логарифмические: одинаковый шаг означает "
    "одинаковое отношение значений. "
    "Суммы не скорректированы на инфляцию.",
    x_log=True,
    y_log=True,
)

scatter_chart(
    rated,
    "rating_clean",
    "revenue_clean",
    "9. Зрительский рейтинг и кассовые сборы",
    "Рейтинг",
    "Сборы, долларов США",
    "Бюджеты и кассовые сборы",
    f"Включены фильмы минимум с {MIN_VOTES} голосами "
    "и положительными сборами. Ось сборов логарифмическая. "
    "Связь показателей не доказывает причинность.",
    y_log=True,
)

leaders = (
    df.dropna(subset=["revenue_clean"])
    .nlargest(10, "revenue_clean")
    .copy()
)

leader_labels = (
    leaders["title"].fillna("Без названия")
    + " [ID "
    + leaders["id"].astype(str)
    + "]"
)

bar_chart(
    pd.Series(
        leaders["revenue_clean"].to_numpy() / 1_000_000,
        index=leader_labels,
    ),
    "10. Фильмы с наибольшими кассовыми сборами",
    "Сборы, млн долларов США",
    "Бюджеты и кассовые сборы",
    "Рейтинг составлен по указанным в датасете сборам. "
    "Это исторический снимок данных, а не актуальный мировой топ. "
    "Кассовые сборы не равны прибыли студии.",
)


# ============================================================
# 11. Описания графиков и выводы
# ============================================================

def number(value, digits=0):
    """Числа с пробелами между тысячами и десятичной запятой."""
    return f"{value:,.{digits}f}".replace(",", " ").replace(".", ",")


# Корреляция Спирмена через ранги.
# Для этого расчёта не нужна дополнительная библиотека scipy.
budget_revenue_corr = (
    financial[["budget_clean", "revenue_clean"]]
    .rank()
    .corr()
    .iloc[0, 1]
)

rating_financial = rated.dropna(
    subset=["rating_clean", "revenue_clean"]
).copy()

rating_revenue_corr = (
    rating_financial[["rating_clean", "revenue_clean"]]
    .rank()
    .corr()
    .iloc[0, 1]
)

peak_year = int(year_counts.idxmax())
peak_count = int(year_counts.max())

language_counts = df["original_language"].value_counts()
english_count = int(language_counts.get("en", 0))
english_share = english_count / len(df) * 100

rating_median = rated["rating_clean"].median()
rating_q1 = rated["rating_clean"].quantile(0.25)
rating_q3 = rated["rating_clean"].quantile(0.75)

runtime_values = df["runtime_clean"].dropna()
runtime_median = runtime_values.median()
runtime_q1 = runtime_values.quantile(0.25)
runtime_q3 = runtime_values.quantile(0.75)

genre_names = {
    "Action": "боевик",
    "Adventure": "приключения",
    "Animation": "мультфильм",
    "Comedy": "комедия",
    "Crime": "криминал",
    "Documentary": "документальное кино",
    "Drama": "драма",
    "Family": "семейное кино",
    "Fantasy": "фэнтези",
    "Foreign": "зарубежное кино",
    "History": "историческое кино",
    "Horror": "ужасы",
    "Music": "музыкальное кино",
    "Mystery": "детектив",
    "Romance": "мелодрама",
    "Science Fiction": "научная фантастика",
    "TV Movie": "телевизионный фильм",
    "Thriller": "триллер",
    "War": "военное кино",
    "Western": "вестерн",
}


def genre_name(value):
    return genre_names.get(value, value)


top_genres_text = "; ".join(
    f"{genre_name(name)} — {number(count)}"
    for name, count in genre_counts.head(3).items()
)

top_directors_text = "; ".join(
    f"{name} — {number(count)}"
    for name, count in director_counts.head(3).items()
)

# Учитываем, что первое место могут делить несколько жанров.
best_genre_rating = genre_rating["median"].max()

best_genres = genre_rating[
    genre_rating["median"].eq(best_genre_rating)
]

best_genres_text = "; ".join(
    f"{genre_name(name)} ({number(row['count'])} фильмов)"
    for name, row in best_genres.iterrows()
)

first_film = leaders.iloc[0]
second_film = leaders.iloc[1]

financial_share = len(financial) / len(df) * 100


chart_texts = [
    {
        "title": "1. Сколько фильмов приходится на каждый год",
        "description": (
            "По горизонтали указан год выхода, по вертикали - "
            "число фильмов этого года в нашей таблице."
        ),
        "conclusion": (
            f"Больше всего фильмов относится к {peak_year} году - "
            f"{number(peak_count)}. В таблице заметно больше "
            "современных фильмов, чем старых. Поэтому результаты "
            "работы в большей степени описывают современное кино. "
            "По этому графику нельзя определить, сколько всего "
            "фильмов выпускали в мире: здесь собрана только часть."
        ),
    },
    {
        "title": "2. Какие жанры встречаются чаще",
        "description": (
            "Длина полосы показывает количество фильмов каждого "
            "жанра. Если у фильма несколько жанров, он учитывается "
            "в каждом из них. Названия на графике сохранены "
            "как в исходном файле."
        ),
        "conclusion": (
            f"Чаще всего встречаются: {top_genres_text}. "
            "Таким образом, основную часть данных составляют "
            "драмы, комедии и триллеры. Это говорит о составе "
            "таблицы, но не показывает, какие жанры зрители "
            "любят больше: для этого нужны оценки или другие "
            "данные о предпочтениях."
        ),
    },
    {
        "title": "3. На каких языках сняты фильмы",
        "description": (
            "Показаны десять самых частых языков оригинала. "
            "Например, en - английский, fr - французский, "
            "es - испанский. Язык оригинала может отличаться "
            "от языка перевода, на котором мы смотрим фильм."
        ),
        "conclusion": (
            f"На английском языке снято {number(english_count)} "
            f"фильмов - {number(english_share, 1)}% всей таблицы. "
            "Фильмов на других языках значительно меньше. "
            "Поэтому общие результаты в основном относятся "
            "к англоязычному кино. Сравнивать по ним киноиндустрию "
            "разных стран было бы некорректно."
        ),
    },
    {
        "title": "4. Чьи фильмы чаще встречаются в таблице",
        "description": (
            "Показаны десять режиссёров с наибольшим количеством "
            "фильмов в данных. При совместной работе фильм "
            "учитывается у каждого его режиссёра."
        ),
        "conclusion": (
            f"Первые три позиции занимают: {top_directors_text}. "
            "Их работы чаще других представлены в этой таблице. "
            "Это не полная фильмография режиссёров и не оценка "
            "качества их фильмов."
        ),
    },
    {
        "title": "5. Какие оценки зрители ставят фильмам",
        "description": (
            f"На графике {number(len(rated))} фильмов, "
            f"за которые проголосовали хотя бы {MIN_VOTES} человек. "
            "Каждый столбец показывает, сколько фильмов попало "
            "в определённый диапазон оценок."
        ),
        "conclusion": (
            f"Медианная оценка составляет "
            f"{number(rating_median, 1)} из 10. "
            "Медиана - середина списка оценок, если расположить "
            "их по порядку. "
            f"Средняя половина оценок находится между "
            f"{number(rating_q1, 1)} и {number(rating_q3, 1)}. "
            "Значит, оценки чаще сосредоточены в средней части "
            "шкалы, а не возле её крайних значений."
        ),
    },
    {
        "title": "6. Сколько обычно длится фильм",
        "description": (
            "По горизонтали указана продолжительность в минутах, "
            "по вертикали - количество фильмов. Записи без "
            "известной положительной продолжительности "
            "не включены в этот график."
        ),
        "conclusion": (
            f"Медианная продолжительность - "
            f"{number(runtime_median)} минуты. "
            f"Средняя половина фильмов длится от "
            f"{number(runtime_q1)} до {number(runtime_q3)} минут. "
            "Это даёт более полезное представление об обычной "
            "длительности фильма, чем самые короткие "
            "или самые длинные записи."
        ),
    },
    {
        "title": "7. Как различаются оценки разных жанров",
        "description": (
            "Для каждого жанра показана медиана зрительских "
            f"оценок. В расчёт входят фильмы минимум с {MIN_VOTES} "
            "голосами. Жанры, в которых осталось меньше "
            "30 таких фильмов, не показаны."
        ),
        "conclusion": (
            f"Самая высокая медианная оценка - "
            f"{number(best_genre_rating, 2)}. "
            f"Она получилась у категории: {best_genres_text}. "
            "Но количество фильмов в жанрах различается. "
            "Высокая оценка небольшой группы ещё не означает, "
            "что любой фильм этого жанра будет оценён лучше. "
            "Этот график сравнивает оценки уже попавших в таблицу "
            "фильмов."
        ),
    },
    {
        "title": "8. Связаны ли бюджет и кассовые сборы",
        "description": (
            f"Каждая точка - один из {number(len(financial))} "
            "фильмов с известными положительными бюджетом "
            "и сборами. Чем правее точка, тем больше бюджет; "
            "чем выше - тем больше сборы. Шкалы показывают рост "
            "в несколько раз, чтобы на одном графике были видны "
            "и небольшие, и крупные суммы."
        ),
        "conclusion": (
            "В этих данных более дорогие фильмы обычно имеют "
            "более высокие сборы. "
            f"Коэффициент корреляции Спирмена равен "
            f"{number(budget_revenue_corr, 3)}. "
            "Он сравнивает порядок фильмов по двум показателям: "
            "значение ближе к 1 означает более выраженную "
            "положительную связь. Однако большой бюджет "
            "не гарантирует высоких сборов. По этим данным "
            "также нельзя определить, какая часть связи "
            "объясняется рекламой или масштабом проката."
        ),
    },
    {
        "title": "9. Собирают ли высокооценённые фильмы больше денег",
        "description": (
            f"Показаны {number(len(rating_financial))} фильмов "
            f"минимум с {MIN_VOTES} голосами и известными "
            "положительными сборами. По горизонтали - оценка, "
            "по вертикали - сборы. Для сборов используется "
            "такая же шкала, как на предыдущем графике."
        ),
        "conclusion": (
            f"Коэффициент корреляции Спирмена составляет "
            f"{number(rating_revenue_corr, 3)} - это значение "
            "близко к нулю. Выраженной связи между оценкой "
            "и сборами в этой группе не видно. Фильмы "
            "с похожими оценками могут собирать совершенно "
            "разные суммы. Поэтому одной зрительской оценки "
            "недостаточно, чтобы судить о кассовом успехе."
        ),
    },
    {
        "title": "10. Какие фильмы собрали больше всего",
        "description": (
            "Показаны десять фильмов с самыми высокими сборами "
            "в исходных данных. Суммы указаны в миллионах "
            "долларов. ID рядом с названием - номер фильма "
            "в базе."
        ),
        "conclusion": (
            f"Первое место занимает «{first_film['title']}» - "
            f"{number(first_film['revenue_clean'] / 1_000_000, 1)} "
            "млн долларов. "
            f"На втором месте «{second_film['title']}» - "
            f"{number(second_film['revenue_clean'] / 1_000_000, 1)} "
            "млн долларов. Это сравнение по значениям "
            "в данном наборе, а не сегодняшний мировой рейтинг. "
            "Большие сборы также не равны прибыли: "
            "затраты и распределение выручки здесь не учтены."
        ),
    },
]

if len(charts) != len(chart_texts):
    raise ValueError(
        "Количество графиков не совпадает с количеством описаний."
    )

for chart, text in zip(charts, chart_texts):
    chart["plot"].title.text = text["title"]
    chart["explanation"] = text["description"]
    chart["conclusion"] = text["conclusion"]


conclusions = [
    (
        f"В работе рассмотрено {number(len(df))} фильмов. "
        "В таблице преобладает англоязычное кино, "
        "поэтому результаты нельзя переносить "
        "на всё мировое кино."
    ),
    (
        "Среди жанров чаще всего встречаются драмы, "
        "комедии и триллеры. Само количество фильмов "
        "не показывает предпочтения зрителей."
    ),
    (
        f"Медианная оценка составляет "
        f"{number(rating_median, 1)} из 10, "
        f"а медианная продолжительность - "
        f"{number(runtime_median)} минуты."
    ),
    (
        "Связь бюджета со сборами оказалась заметно сильнее, "
        "чем связь зрительской оценки со сборами. "
        "Большой бюджет часто сопровождается высокими сборами, "
        "но ни бюджет, ни оценка не гарантируют успеха."
    ),
    (
        f"Одновременно бюджет и сборы известны для "
        f"{number(len(financial))} фильмов "
        f"({number(financial_share, 1)}% таблицы). "
        "Финансовые выводы сделаны только по этим записям. "
        "Добавление недостающих данных могло бы "
        "изменить результаты."
    ),
]


# ============================================================
# 12. Фотография автора
# ============================================================

if PHOTO_PATH.exists():
    encoded_photo = base64.b64encode(
        PHOTO_PATH.read_bytes()
    ).decode("ascii")

    photo_html = (
        '<img class="photo" alt="Фото автора" '
        f'src="data:image/jpeg;base64,{encoded_photo}">'
    )
else:
    photo_html = (
        '<div class="photo placeholder">Фото автора</div>'
    )


# ============================================================
# 13. Сборка страницы
# ============================================================

script, plot_divs = components(
    [item["plot"] for item in charts]
)

group_descriptions = {
    "Состав выборки": (
        "Какие фильмы попали в таблицу",
        "Сначала посмотрим на годы выхода, жанры, языки "
        "и режиссёров. Это поможет понять, "
        "какие фильмы мы сравниваем.",
    ),
    "Рейтинги и продолжительность": (
        "Оценки и продолжительность",
        "Посмотрим, какие оценки встречаются чаще, "
        "сколько обычно длится фильм и как оценки "
        "различаются по жанрам.",
    ),
    "Бюджеты и кассовые сборы": (
        "Бюджеты и сборы",
        "Проверим, связаны ли сборы с бюджетом "
        "и зрительской оценкой. Затем посмотрим "
        "на самые кассовые фильмы в таблице.",
    ),
}

sections = []
current_group = None

for item, plot_div in zip(charts, plot_divs):
    group = item["group"]

    if group != current_group:
        heading, description = group_descriptions[group]

        sections.append(
            f"<h2>{html.escape(heading)}</h2>"
            f"<p>{html.escape(description)}</p>"
        )

        current_group = group

    sections.append(
        '<article class="card">'
        f'{plot_div}'
        '<p class="chart-description">'
        f'{html.escape(item["explanation"])}'
        '</p>'
        '<div class="chart-conclusion">'
        '<b>Вывод</b>'
        f'<p>{html.escape(item["conclusion"])}</p>'
        '</div>'
        '</article>'
    )


# Краткая таблица проверок.
# Полный технический отчёт остаётся в quality_report.json.
quality_rows = [
    ("Фильмов в исходном файле", len(movies)),
    ("Записей в файле об актёрах и съёмочной группе", len(credits)),
    ("Полных повторов в файле фильмов",
     quality["Полные дубликаты movies"]),
    ("Полных повторов в файле съёмочной группы",
     quality["Полные дубликаты credits"]),
    ("Фильмов без соответствующей записи о съёмочной группе",
     quality["Фильмы без соответствия в credits"]),
    ("Пустых или некорректных дат выхода",
     quality["Пустые или некорректные даты выхода"]),
    ("Фильмов с нулём в поле бюджета",
     quality["Нулевые значения: budget"]),
    ("Фильмов с нулём в поле сборов",
     quality["Нулевые значения: revenue"]),
    ("Фильмов с нулём в поле продолжительности",
     quality["Нулевые значения: runtime"]),
    ("Фильмов без указанного жанра",
     quality["Фильмы без указанного жанра"]),
    ("Фильмов после обработки", len(df)),
    ("Фильмов для сравнения бюджета и сборов", len(financial)),
    (f"Фильмов с оценкой и минимум {MIN_VOTES} голосами", len(rated)),
]

quality_table = pd.DataFrame(
    quality_rows,
    columns=["Что проверено", "Количество"],
).to_html(
    index=False,
    border=0,
    classes="data-table",
)


column_names = {
    "budget": "Бюджет",
    "genres": "Жанры",
    "homepage": "Сайт фильма",
    "id": "Номер фильма",
    "keywords": "Ключевые слова",
    "original_language": "Язык оригинала",
    "original_title": "Оригинальное название",
    "overview": "Описание",
    "popularity": "Показатель популярности",
    "production_companies": "Кинокомпании",
    "production_countries": "Страны производства",
    "release_date": "Дата выхода",
    "revenue": "Кассовые сборы",
    "runtime": "Продолжительность",
    "spoken_languages": "Языки в фильме",
    "status": "Статус выпуска",
    "tagline": "Слоган",
    "title": "Название",
    "vote_average": "Зрительская оценка",
    "vote_count": "Число голосов",
}

missing_rows = [
    (column_names.get(column, column), count)
    for column, count in initial_missing.items()
    if count > 0
]

missing_table = pd.DataFrame(
    missing_rows,
    columns=["Поле", "Пустых значений"],
).to_html(
    index=False,
    border=0,
    classes="data-table",
)


language_names = {
    "en": "английский",
    "fr": "французский",
    "es": "испанский",
    "de": "немецкий",
    "zh": "китайский",
    "ja": "японский",
    "hi": "хинди",
    "it": "итальянский",
    "ko": "корейский",
    "ru": "русский",
    "pt": "португальский",
    "cn": "кантонский",
}

preview_df = df[
    [
        "title",
        "release_year",
        "original_language",
        "rating_clean",
        "budget_clean",
        "revenue_clean",
    ]
].head(12).copy()

preview_df["original_language"] = (
    preview_df["original_language"].map(
        lambda value: language_names.get(value, value)
    )
)

preview_df = preview_df.rename(columns={
    "title": "Фильм",
    "release_year": "Год",
    "original_language": "Язык",
    "rating_clean": "Оценка",
    "budget_clean": "Бюджет, $",
    "revenue_clean": "Сборы, $",
})

preview = preview_df.to_html(
    index=False,
    border=0,
    classes="data-table",
    na_rep="Нет данных",
    float_format=lambda value: number(value, 2),
)

conclusions_html = "".join(
    f"<li>{html.escape(text)}</li>"
    for text in conclusions
)


page = f"""<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Анализ фильмов TMDB</title>
    {INLINE.render()}
       <style>
        * {{
            box-sizing: border-box;
        }}

        html {{
            scroll-behavior: smooth;
            scroll-padding-top: 24px;
        }}

        body {{
            margin: 0;
            font-family: Arial, Helvetica, sans-serif;
            background: #F3F3F3;
            color: #111111;
            line-height: 1.65;
        }}

        main {{
            max-width: 1200px;
            margin: 0 auto;
            padding: 28px 20px 64px;
        }}

        header {{
            display: flex;
            align-items: center;
            gap: 28px;
            padding: 36px;
            margin-bottom: 24px;
            background: #171717;
            color: #FFFFFF;
            border-top: 5px solid #FF6600;
            border-radius: 18px;
        }}

        header h1 {{
            margin: 12px 0 18px;
            color: #FFFFFF;
            font-size: clamp(28px, 4vw, 42px);
            line-height: 1.15;
            letter-spacing: -1px;
        }}

        header p {{
            margin: 10px 0;
            color: #E0E0E0;
        }}

        header > div > p:first-child {{
            color: #FF994F;
            font-size: 13px;
            font-weight: bold;
            letter-spacing: 0.5px;
        }}

        header b {{
            color: #FFFFFF;
        }}

        h2 {{
            margin: 36px 0 16px;
            color: #111111;
            font-size: 27px;
            line-height: 1.3;
        }}

        h3 {{
            margin: 26px 0 12px;
            color: #111111;
            font-size: 20px;
            line-height: 1.4;
        }}

        p {{
            margin: 12px 0;
        }}

        a {{
            color: #A83D00;
            text-underline-offset: 3px;
        }}

        a:hover {{
            color: #762B00;
        }}

        a:focus-visible,
        summary:focus-visible {{
            outline: 3px solid #FF6600;
            outline-offset: 4px;
            border-radius: 4px;
        }}

        nav {{
            display: flex;
            flex-wrap: wrap;
            gap: 12px;
            margin: 0 0 28px;
        }}

        nav a {{
            display: inline-flex;
            align-items: center;
            justify-content: center;
            min-height: 46px;
            padding: 11px 22px;
            background: #FF6600;
            color: #111111;
            border: 1px solid #FF6600;
            border-radius: 9px;
            font-size: 15px;
            font-weight: bold;
            text-decoration: none;
            transition:
                background 0.15s ease,
                border-color 0.15s ease;
        }}

        nav a:hover {{
            background: #FF8533;
            color: #111111;
            border-color: #FF8533;
        }}

        .card {{
            padding: 28px;
            margin-bottom: 24px;
            background: #FFFFFF;
            border: 1px solid #E5E5E5;
            border-radius: 16px;
            box-shadow: 0 4px 16px rgb(0 0 0 / 4%);
        }}

        .card > h2:first-child {{
            margin-top: 0;
        }}

        .photo {{
            width: 140px;
            height: 170px;
            object-fit: cover;
            border: 3px solid #FF6600;
            border-radius: 12px;
            flex-shrink: 0;
        }}

        .placeholder {{
            display: grid;
            place-content: center;
            background: #292929;
            color: #FFFFFF;
            text-align: center;
        }}

        .table-wrap {{
            overflow-x: auto;
            margin-top: 16px;
            border: 1px solid #E5E5E5;
            border-radius: 10px;
        }}

        .data-table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
            color: #111111;
        }}

        .data-table th,
        .data-table td {{
            padding: 13px 15px;
            border-bottom: 1px solid #EAEAEA;
            text-align: left;
            vertical-align: top;
        }}

        .data-table th {{
            background: #202020;
            color: #FFFFFF;
            font-weight: bold;
        }}

        .data-table tbody tr:nth-child(odd) {{
            background: #FFFFFF;
        }}

        .data-table tbody tr:nth-child(even) {{
            background: #FAFAFA;
        }}

        .data-table tbody tr:hover {{
            background: #FFF0E5;
        }}

        .data-table tbody tr:last-child td {{
            border-bottom: none;
        }}

        ol,
        ul {{
            padding-left: 24px;
        }}

        li {{
            margin-bottom: 14px;
            padding-left: 4px;
        }}

        li::marker {{
            color: #A83D00;
            font-weight: bold;
        }}

        .note {{
            margin: 20px 0;
            padding: 18px 20px;
            background: #FFF0E5;
            color: #111111;
            border-left: 4px solid #FF6600;
            border-radius: 8px;
        }}

        .chart-description {{
            margin: 20px 0;
            color: #333333;
        }}

        .chart-conclusion {{
            margin-top: 18px;
            padding: 20px 22px;
            background: #FFF0E5;
            color: #111111;
            border-left: 5px solid #FF6600;
            border-radius: 10px;
        }}

        .chart-conclusion > b {{
            display: block;
            color: #A83D00;
            font-size: 18px;
        }}

        .chart-conclusion p {{
            margin: 8px 0 0;
        }}

        details {{
            margin-top: 22px;
            padding: 16px 18px;
            background: #FAFAFA;
            border: 1px solid #E5E5E5;
            border-radius: 10px;
        }}

        summary {{
            cursor: pointer;
            color: #111111;
            font-weight: bold;
        }}

        summary::marker {{
            color: #A83D00;
        }}

        details[open] summary {{
            margin-bottom: 14px;
        }}

        code {{
            padding: 2px 5px;
            background: #F0F0F0;
            color: #111111;
            border-radius: 4px;
            font-size: 0.92em;
            overflow-wrap: anywhere;
        }}

        #conclusions {{
            border-top: 4px solid #FF6600;
        }}

        footer {{
            padding-top: 16px;
            color: #555555;
            font-size: 13px;
        }}

        @media (max-width: 600px) {{
            main {{
                padding: 14px 12px 40px;
            }}

            header {{
                flex-direction: column;
                align-items: flex-start;
                gap: 16px;
                padding: 24px 20px;
            }}

            header h1 {{
                font-size: 29px;
                letter-spacing: -0.5px;
            }}

            .photo {{
                width: 110px;
                height: 135px;
            }}

            h2 {{
                font-size: 23px;
            }}

            .card {{
                padding: 16px;
                border-radius: 12px;
            }}

            nav {{
                gap: 8px;
            }}

            nav a {{
                flex: 1 1 140px;
                padding: 10px 14px;
                font-size: 14px;
            }}

            .chart-conclusion {{
                padding: 16px;
            }}

            .data-table th,
            .data-table td {{
                padding: 10px;
            }}
        }}

        @media (prefers-reduced-motion: reduce) {{
            html {{
                scroll-behavior: auto;
            }}

            nav a {{
                transition: none;
            }}
        }}
    </style>
</head>
<body>
<main>
    <header>
        {photo_html}
        <div>
            <p>Индивидуальная работа по дисциплине
            «Системы больших данных»</p>
            <h1>Какие фильмы смотрят и сколько они собирают в прокате</h1>
            <p>Анализ жанров, оценок, бюджетов и кассовых сборов
            по данным TMDB</p>
            <p>
                <b>Выполнил:</b> {html.escape(AUTHOR)}<br>
                <b>Группа:</b> {html.escape(GROUP)}
            </p>
        </div>
    </header>

    <nav>
        <a href="#dataset">О работе</a>
        <a href="#cleaning">Подготовка данных</a>
        <a href="#charts">Графики</a>
        <a href="#conclusions">Итоги</a>
    </nav>

    <section id="dataset" class="card">
        <h2>Что я исследовал</h2>

        <p>В этой работе я сравнил фильмы по жанрам,
        зрительским оценкам, продолжительности, бюджету
        и кассовым сборам.</p>

        <p>Главные вопросы: какие фильмы чаще встречаются
        в данных, какие оценки они получают и связаны ли
        высокие сборы с большим бюджетом или хорошей оценкой.</p>

        <p>Для работы использован набор
        <a href="{SOURCE_URL}">TMDB 5000 Movie Dataset с Kaggle</a>.
        После обработки в таблице осталось
        {number(len(df))} фильмов.</p>

        <p>В файле <code>tmdb_5000_movies.csv</code>
        находятся сведения о фильмах,
        а в <code>tmdb_5000_credits.csv</code> -
        об актёрах и съёмочной группе.
        Я объединил файлы по номеру фильма,
        чтобы добавить сведения о режиссёрах.</p>

        <p class="note">В набор входят не все существующие фильмы.
        Оценки и сборы приведены на момент сбора данных
        и могли измениться. Поэтому все выводы ниже относятся
        к этой таблице.</p>

        <h3>Пример данных после обработки</h3>

        <p>Ниже показаны первые 12 записей.
        Бюджеты и сборы указаны в долларах США.</p>

        <div class="table-wrap">{preview}</div>
    </section>

    <section id="cleaning" class="card">
        <h2>Как я подготовил данные</h2>

        <p>Перед построением графиков я проверил таблицы
        и подготовил поля для расчётов.</p>

        <ol>
            <li>
                <b>Проверил повторы и номера фильмов.</b>
                Полностью одинаковые записи и пустые строки
                удаляются. У каждого фильма должен быть свой
                номер, чтобы при объединении файлов
                записи не дублировались.
            </li>

            <li>
                <b>Привёл текст к одному виду.</b>
                Убрал лишние пробелы в начале и конце,
                заменил неразрывные пробелы обычными
                и унифицировал запись символов Unicode.
            </li>

            <li>
                <b>Проверил числа и даты.</b>
                Числовые поля преобразовал в числа,
                дату выхода - в дату.
                Некорректные и отрицательные значения
                в проверяемых числовых полях
                исключил из расчётов.
            </li>

            <li>
                <b>Отдельно обработал нули.</b>
                Для этой работы нулевой бюджет, сборы
                или продолжительность считаются отсутствием
                пригодных данных. Такие значения не участвуют
                в соответствующих графиках, но сам фильм
                остаётся в общей таблице.
            </li>

            <li>
                <b>Извлёк жанры и режиссёров.</b>
                В исходных файлах они записаны списками
                в формате JSON. Я преобразовал их так,
                чтобы можно было посчитать фильмы
                каждого жанра и режиссёра.
            </li>

            <li>
                <b>Нормализовал числовые показатели.</b>
                Для бюджета, сборов, продолжительности,
                оценки и популярности создал дополнительные
                столбцы со значениями от 0 до 1.
            </li>
        </ol>

        <h3>Что означает нормализация</h3>

        <p>Бюджет измеряется миллионами долларов,
        продолжительность - минутами, а оценка -
        баллами от 0 до 10. Нормализация переводит
        эти разные шкалы в общий диапазон.</p>

        <p class="note">
            Формула:
            <code>x_norm = (x − min) / (max − min)</code>.<br>
            Минимальное значение становится 0,
            максимальное - 1, остальные располагаются между ними.
        </p>

        <p>Нормализованные значения сохранены в подготовленном
        CSV-файле. На графиках оставлены доллары, минуты
        и баллы, чтобы результаты было проще читать.
        Пустые значения не заменялись средними.</p>

        <p>Очень большие положительные значения
        не удалялись автоматически: высокая стоимость
        фильма или большая продолжительность сами по себе
        ещё не означают ошибку.</p>

        <h3>Что показала проверка</h3>
        <div class="table-wrap">{quality_table}</div>

        <details>
            <summary>В каких полях были пропуски</summary>
            <p>В этой таблице показаны только поля
            с пустыми значениями в исходном файле.
            Нули и пустые списки здесь не считаются
            пропусками - они проверяются отдельно.</p>
            <div class="table-wrap">{missing_table}</div>
        </details>
    </section>

    <section id="charts">
        <h2>Результаты анализа</h2>

        <p>Наведите курсор на точку или столбец,
        чтобы увидеть точные значения.
        На панели каждого графика можно изменить масштаб,
        вернуть исходный вид или сохранить изображение.</p>

        {''.join(sections)}
    </section>

    <section id="conclusions" class="card">
        <h2>Что удалось выяснить</h2>

        <ol>{conclusions_html}</ol>

        <h3>Что нужно учитывать</h3>

        <p>В таблице неравномерно представлены языки,
        жанры и годы выпуска. Некоторые фильмы не имеют
        сведений о бюджете или сборах. Поэтому результаты
        могут отличаться от анализа другого набора фильмов.</p>

        <p>Суммы разных лет сравниваются без поправки
        на изменение цен. Кроме того, кассовые сборы
        не показывают чистую прибыль студии.</p>

        <p>Найденные связи помогают описать данные,
        но не объясняют все причины успеха фильма.
        Например, в этой работе отдельно не проверялось
        влияние рекламы, известности актёров
        или количества кинотеатров.</p>

    </section>
</main>

{script}
</body>
</html>
"""

result_path = OUTPUT_DIR / "index.html"
result_path.write_text(page, encoding="utf-8")

print("Готово.")
print(f"Фильмов после обработки: {len(df)}")
print(f"Графиков: {len(charts)}")
print(f"Страница: {result_path}")
print("Откройте output/index.html в браузере.")