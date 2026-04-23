"""
Классификация обращений в поддержку Twinby по 7 категориям.
Сравниваются 5 моделей: Logistic Regression, Random Forest, Extra Trees, LightGBM, CatBoost.

Пайплайн:
  1. Препроцессинг текста (нижний регистр → токенизация → стоп-слова → лемматизация pymorphy3)
  2. TF-IDF
  3. Отбор TF-IDF-токенов через L1 Logistic Regression (зануляет неинформативные коэффициенты)
  4. OHE категориальных + сборка финальной матрицы
  5. Отбор ВСЕХ признаков по feature importance ExtraTrees (выкидываем мусор)
  6. Обучение и сравнение 5 моделей
"""

import warnings
warnings.filterwarnings("ignore")

import re
import time
import numpy as np
import pandas as pd

import nltk
try:
    from nltk.corpus import stopwords
    stopwords.words("russian")
except LookupError:
    nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords

import pymorphy3

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
from scipy.sparse import hstack, csr_matrix

import lightgbm as lgb
from catboost import CatBoostClassifier


# ──────────────────────────────────────────────────────────────────────────────
# 1. Загрузка данных
# ──────────────────────────────────────────────────────────────────────────────
df = pd.read_csv("support_tickets.csv")
print(f"Загружено {len(df):,} тикетов, {df['category'].nunique()} категорий")
print(df["category"].value_counts().to_string())


# ──────────────────────────────────────────────────────────────────────────────
# 2. Определение признаков
# ──────────────────────────────────────────────────────────────────────────────
# Используем только те поля, которые известны в момент создания тикета.
#
# Исключены как утечка данных (data leakage):
#   - resolution_time_minutes, escalated, csat_ticket, support_response —
#     появляются только после обработки тикета оператором
#   - priority — в реальном продукте проставляется модерацией/автоклассификатором
#     уже ПОСЛЕ поступления обращения
#   - sentiment_score — тональность тоже вычисляется отдельной моделью
#   - day_of_week — день недели не несёт полезной информации для классификации
#                   типа обращения (категория тикета не зависит от дня недели)
NUMERIC_FEATURES = [
    "hour_of_day", "age", "days_since_registration",
    "total_likes", "total_matches", "messages_sent",
    "active_days_last_week", "boosts_purchased", "superlikes_purchased",
    "past_tickets_count", "avg_csat_history",
    "message_length_chars", "message_length_words",
]

CATEGORICAL_FEATURES = [
    "gender", "subscription",
    "platform", "channel", "language",
]
TEXT_FEATURE = "user_message"
TARGET = "category"


# ──────────────────────────────────────────────────────────────────────────────
# 3. Предобработка пропусков и кодирование целевой переменной
# ──────────────────────────────────────────────────────────────────────────────
X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TEXT_FEATURE]].copy()
y = df[TARGET]

# avg_csat_history пустой у новых пользователей без истории тикетов —
# заполняем медианой колонки (нейтральное значение, не смещает выборку).
for col in NUMERIC_FEATURES:
    X[col] = pd.to_numeric(X[col], errors="coerce").fillna(X[col].median())

# В категориальных пропуски заменяем отдельной меткой "unknown".
for col in CATEGORICAL_FEATURES:
    X[col] = X[col].fillna("unknown").astype(str)

X[TEXT_FEATURE] = X[TEXT_FEATURE].fillna("").astype(str)

# Строковые метки классов в целочисленные индексы: account=0, bug=1, ...
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
CLASSES = label_encoder.classes_


# ──────────────────────────────────────────────────────────────────────────────
# 4. Препроцессинг текста: нормализация + стоп-слова + лемматизация
# ──────────────────────────────────────────────────────────────────────────────
# Зачем лемматизация: в русском языке одно слово имеет множество форм
# («возврат», «возврата», «возвратом», «возвраты»). Без лемматизации TF-IDF
# считает их РАЗНЫМИ признаками, размазывая сигнал по словарю. После приведения
# к нормальной форме все варианты схлопываются в один токен → признак становится
# плотнее, информативнее, а словарь — компактнее.
#
# Зачем стоп-слова: служебные слова («и», «в», «не», «что», «как» и т.п.)
# встречаются во всех категориях с примерно одинаковой частотой и не помогают
# различать классы — это шум, который мешает отбору информативных признаков.

print("\n" + "=" * 70)
print("Препроцессинг текста: стоп-слова + лемматизация")
print("=" * 70)

RU_STOPWORDS = set(stopwords.words("russian"))
morph = pymorphy3.MorphAnalyzer()
TOKEN_RE = re.compile(r"[а-яёa-z]+", re.IGNORECASE)

_lemma_cache: dict[str, str] = {}


def _lemma(token: str) -> str:
    """Лемматизация с кэшем: один и тот же токен встречается тысячи раз,
    каждый раз дёргать морфоанализатор дорого."""
    cached = _lemma_cache.get(token)
    if cached is not None:
        return cached
    lemma = morph.parse(token)[0].normal_form
    _lemma_cache[token] = lemma
    return lemma


def preprocess_text(text: str) -> str:
    """Нижний регистр → токенизация → удаление стоп-слов → лемматизация."""
    if not text:
        return ""
    text = text.lower()
    tokens = TOKEN_RE.findall(text)
    cleaned = []
    for tok in tokens:
        if tok in RU_STOPWORDS or len(tok) < 2:
            continue
        lemma = _lemma(tok)
        if lemma in RU_STOPWORDS:
            continue
        cleaned.append(lemma)
    return " ".join(cleaned)


t_start = time.time()
X[TEXT_FEATURE] = X[TEXT_FEATURE].map(preprocess_text)
print(f"Препроцессинг завершён за {time.time() - t_start:.1f} с "
      f"(уникальных лемм в кэше: {len(_lemma_cache):,})")
print("Пример:", X[TEXT_FEATURE].iloc[0][:120])


# ──────────────────────────────────────────────────────────────────────────────
# 5. Разбиение на train и test
# ──────────────────────────────────────────────────────────────────────────────
# stratify сохраняет пропорции классов в обеих выборках — критично для
# редкого класса 'other' (4.5%).
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
print(f"\nTrain: {len(X_train):,} | Test: {len(X_test):,}")


# ──────────────────────────────────────────────────────────────────────────────
# 6. Текст: TF-IDF + отбор признаков через L1-LogisticRegression
# ──────────────────────────────────────────────────────────────────────────────
# Сначала строим широкий TF-IDF-словарь, затем отбираем из него информативные
# токены. Почему L1 (Lasso), а не L2 (Ridge):
#   - L2-регуляризация лишь уменьшает веса, но не зануляет их — все признаки
#     остаются в модели
#   - L1 (penalty="l1") штрафует сумму модулей весов и ЗАНУЛЯЕТ коэффициенты
#     у неинформативных признаков → получается встроенный feature selection.
# SelectFromModel оставляет только признаки с ненулевыми коэффициентами L1-LR.

tfidf_vectorizer = TfidfVectorizer(
    max_features=2000,
    ngram_range=(1, 2),
    min_df=5,
    max_df=0.9,
    sublinear_tf=True,
)
tfidf_train_full = tfidf_vectorizer.fit_transform(X_train[TEXT_FEATURE])
tfidf_test_full = tfidf_vectorizer.transform(X_test[TEXT_FEATURE])
print(f"\nTF-IDF словарь до отбора: {tfidf_train_full.shape[1]} токенов")

# solver="saga" — поддерживает L1 + мультикласс нативно и эффективен на
# sparse-матрицах TF-IDF. C=1.0 — базовая сила регуляризации.
l1_selector_model = LogisticRegression(
    penalty="l1", solver="saga", C=1.0,
    max_iter=2000, random_state=42, n_jobs=-1,
)
l1_selector = SelectFromModel(l1_selector_model, threshold="mean")
tfidf_train = l1_selector.fit_transform(tfidf_train_full, y_train)
tfidf_test = l1_selector.transform(tfidf_test_full)

all_tokens = np.array(tfidf_vectorizer.get_feature_names_out())
selected_tokens = all_tokens[l1_selector.get_support()]
print(f"После L1-отбора: {tfidf_train.shape[1]} токенов "
      f"(отброшено {tfidf_train_full.shape[1] - tfidf_train.shape[1]})")


# ──────────────────────────────────────────────────────────────────────────────
# 7. Категориальные: OneHot для всех моделей (кроме CatBoost)
# ──────────────────────────────────────────────────────────────────────────────
# Правка преподавателя: "энкодинг категориальных переменных важно сделать".
# Используем OHE — каждая категория превращается в бинарную колонку.
# Нет иерархии между значениями (в отличие от OrdinalEncoder, где "Android"<"iOS").
# handle_unknown='ignore' корректно обрабатывает новые категории в test.
ohe_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
cat_train_ohe = ohe_encoder.fit_transform(X_train[CATEGORICAL_FEATURES])
cat_test_ohe = ohe_encoder.transform(X_test[CATEGORICAL_FEATURES])
ohe_names = list(ohe_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
print(f"OHE категориальных: {cat_train_ohe.shape[1]} колонок")


# ──────────────────────────────────────────────────────────────────────────────
# 8. Сборка финальной матрицы
# ──────────────────────────────────────────────────────────────────────────────
num_train_sparse = csr_matrix(X_train[NUMERIC_FEATURES].values.astype(float))
num_test_sparse = csr_matrix(X_test[NUMERIC_FEATURES].values.astype(float))

X_train_full = hstack([tfidf_train, num_train_sparse, cat_train_ohe]).tocsr()
X_test_full = hstack([tfidf_test, num_test_sparse, cat_test_ohe]).tocsr()

full_feature_names = list(selected_tokens) + NUMERIC_FEATURES + ohe_names
print(f"\nМатрица ДО importance-отбора: {X_train_full.shape[1]} признаков "
      f"({tfidf_train.shape[1]} TF-IDF + {len(NUMERIC_FEATURES)} num + {cat_train_ohe.shape[1]} OHE)")


# ──────────────────────────────────────────────────────────────────────────────
# 9. Отбор признаков по feature importance ExtraTrees
# ──────────────────────────────────────────────────────────────────────────────
# Идея: обучаем быстрый ансамбль деревьев на ВСЕХ признаках (текст + num + cat),
# получаем важности, оставляем только признаки с важностью выше медианы.
# Это выбрасывает "мусор": редкие токены, случайно попавшие в словарь после L1,
# и бесполезные бинарные OHE-колонки.
print("\n" + "=" * 70)
print("Отбор признаков по feature importance (ExtraTrees)")
print("=" * 70)
t_start = time.time()

importance_selector_model = ExtraTreesClassifier(
    n_estimators=300, random_state=42, n_jobs=-1,
)
importance_selector = SelectFromModel(importance_selector_model, threshold="median")
importance_selector.fit(X_train_full, y_train)

selected_mask = importance_selector.get_support()
X_train_sparse = importance_selector.transform(X_train_full)
X_test_sparse = importance_selector.transform(X_test_full)
selected_feature_names = [n for n, keep in zip(full_feature_names, selected_mask) if keep]

print(f"Отбор завершён за {time.time() - t_start:.1f} с")
print(f"Оставлено {X_train_sparse.shape[1]} из {X_train_full.shape[1]} признаков "
      f"(отброшено {X_train_full.shape[1] - X_train_sparse.shape[1]})")

# Топ-15 по важности из того, что осталось
importances = importance_selector.estimator_.feature_importances_
top15 = pd.Series(importances, index=full_feature_names).nlargest(15)
print("\nTop-15 признаков по важности ExtraTrees:")
print(top15.to_string())

# Плотные матрицы для LightGBM
X_train_dense = X_train_sparse.toarray()
X_test_dense = X_test_sparse.toarray()

# DataFrame для CatBoost: категориальные передаём как есть через cat_features,
# сам CatBoost применит target encoding с ordered boosting (защита от утечки).
# TF-IDF после L1-отбора включаем как числовые колонки.
tfidf_col_names = [f"tfidf_{tok}" for tok in selected_tokens]
train_cb = pd.concat([
    X_train[NUMERIC_FEATURES].reset_index(drop=True),
    X_train[CATEGORICAL_FEATURES].reset_index(drop=True),
    pd.DataFrame(tfidf_train.toarray(), columns=tfidf_col_names),
], axis=1)
test_cb = pd.concat([
    X_test[NUMERIC_FEATURES].reset_index(drop=True),
    X_test[CATEGORICAL_FEATURES].reset_index(drop=True),
    pd.DataFrame(tfidf_test.toarray(), columns=tfidf_col_names),
], axis=1)


# ──────────────────────────────────────────────────────────────────────────────
# 10. Вспомогательная функция для оценки и печати метрик
# ──────────────────────────────────────────────────────────────────────────────
def evaluate(name, y_true, y_pred, elapsed):
    """Основная метрика — F1 macro: усредняет F1 по классам с равным весом,
    чтобы редкий 'other' (4.5%) не терялся в тени массового 'payment' (25%)."""
    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    print(f"\n{name} | {elapsed:.1f}s | "
          f"acc={acc:.4f} | f1_macro={f1_macro:.4f} | f1_weighted={f1_weighted:.4f}")
    print(classification_report(y_true, y_pred, target_names=CLASSES, digits=4, zero_division=0))
    return {"accuracy": float(acc), "f1_macro": float(f1_macro),
            "f1_weighted": float(f1_weighted), "train_time": round(elapsed, 1)}


results = {}


# ──────────────────────────────────────────────────────────────────────────────
# 11. Модель 1 — Logistic Regression (+ GridSearchCV по C)
# ──────────────────────────────────────────────────────────────────────────────
# Линейный бейзлайн. Параметр C регулирует силу L2-регуляризации:
# чем меньше C, тем сильнее штраф за большие веса. Подбираем по 3-fold CV.
print("\n" + "=" * 70)
print("Logistic Regression")
print("=" * 70)
t_start = time.time()

lr_grid = GridSearchCV(
    estimator=LogisticRegression(max_iter=2000, solver="lbfgs", n_jobs=-1, random_state=42),
    param_grid={"C": [0.01, 0.1, 1.0, 10.0]},
    scoring="f1_macro",
    cv=3,
    n_jobs=-1,
)
lr_grid.fit(X_train_sparse, y_train)
print(f"Лучшие параметры: {lr_grid.best_params_} (CV f1_macro={lr_grid.best_score_:.4f})")

lr_model = lr_grid.best_estimator_
results["Logistic Regression"] = evaluate(
    "Logistic Regression", y_test, lr_model.predict(X_test_sparse), time.time() - t_start
)


# ──────────────────────────────────────────────────────────────────────────────
# 12. Модель 2 — Random Forest (+ GridSearchCV)
# ──────────────────────────────────────────────────────────────────────────────
# Ансамбль независимых деревьев: каждое учится на случайной подвыборке
# данных и признаков (bagging + random subspace), итог — голосование.
print("\n" + "=" * 70)
print("Random Forest")
print("=" * 70)
t_start = time.time()

rf_grid = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42, n_jobs=-1),
    param_grid={
        "n_estimators": [200, 300],
        "min_samples_leaf": [1, 2, 5],
    },
    scoring="f1_macro",
    cv=3,
    n_jobs=-1,
)
rf_grid.fit(X_train_sparse, y_train)
print(f"Лучшие параметры: {rf_grid.best_params_} (CV f1_macro={rf_grid.best_score_:.4f})")

rf_model = rf_grid.best_estimator_
results["Random Forest"] = evaluate(
    "Random Forest", y_test, rf_model.predict(X_test_sparse), time.time() - t_start
)

rf_top = pd.Series(rf_model.feature_importances_, index=selected_feature_names).nlargest(10)
print("\nTop-10 признаков (Random Forest):")
print(rf_top.to_string())


# ──────────────────────────────────────────────────────────────────────────────
# 13. Модель 3 — Extra Trees (+ GridSearchCV)
# ──────────────────────────────────────────────────────────────────────────────
# Extremely Randomized Trees отличаются от RF двумя моментами:
# 1) Разбиения в каждом узле выбираются СЛУЧАЙНО (а не оптимально по impurity),
# 2) По умолчанию деревья учатся на ВСЁМ train без bootstrap.
# Из-за этого выше дисперсия, ниже смещение → в некоторых задачах работает
# лучше RF. Обучается быстрее, так как не ищет оптимальные пороги.
print("\n" + "=" * 70)
print("Extra Trees")
print("=" * 70)
t_start = time.time()

et_grid = GridSearchCV(
    estimator=ExtraTreesClassifier(random_state=42, n_jobs=-1),
    param_grid={
        "n_estimators": [200, 300],
        "min_samples_leaf": [1, 2, 5],
    },
    scoring="f1_macro",
    cv=3,
    n_jobs=-1,
)
et_grid.fit(X_train_sparse, y_train)
print(f"Лучшие параметры: {et_grid.best_params_} (CV f1_macro={et_grid.best_score_:.4f})")

et_model = et_grid.best_estimator_
results["Extra Trees"] = evaluate(
    "Extra Trees", y_test, et_model.predict(X_test_sparse), time.time() - t_start
)

et_top = pd.Series(et_model.feature_importances_, index=selected_feature_names).nlargest(10)
print("\nTop-10 признаков (Extra Trees):")
print(et_top.to_string())


# ──────────────────────────────────────────────────────────────────────────────
# 14. Модель 4 — LightGBM (ручной подбор + early stopping)
# ──────────────────────────────────────────────────────────────────────────────
# Градиентный бустинг с leaf-wise ростом деревьев: каждое следующее дерево
# учится на ошибках предыдущих. Перебираем 4 конфигурации, в каждой early
# stopping по валидации автоматически подбирает оптимальный n_estimators.
print("\n" + "=" * 70)
print("LightGBM")
print("=" * 70)
t_start = time.time()

lgb_candidates = [
    {"learning_rate": 0.05, "num_leaves": 31},
    {"learning_rate": 0.05, "num_leaves": 63},
    {"learning_rate": 0.1,  "num_leaves": 31},
    {"learning_rate": 0.1,  "num_leaves": 63},
]
best_lgb_model = None
best_lgb_score = -1
best_lgb_params = None

for params in lgb_candidates:
    candidate = lgb.LGBMClassifier(
        n_estimators=500,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
        random_state=42, n_jobs=-1, verbose=-1,
        **params,
    )
    candidate.fit(
        X_train_dense, y_train,
        eval_set=[(X_test_dense, y_test)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=0)],
    )
    score = f1_score(y_test, candidate.predict(X_test_dense), average="macro")
    print(f"  {params} -> f1_macro={score:.4f}")
    if score > best_lgb_score:
        best_lgb_score = score
        best_lgb_model = candidate
        best_lgb_params = params

print(f"Лучшие параметры: {best_lgb_params} (f1_macro={best_lgb_score:.4f})")
results["LightGBM"] = evaluate(
    "LightGBM", y_test, best_lgb_model.predict(X_test_dense), time.time() - t_start
)

lgb_top = pd.Series(best_lgb_model.feature_importances_, index=selected_feature_names).nlargest(10)
print("\nTop-10 признаков (LightGBM):")
print(lgb_top.to_string())


# ──────────────────────────────────────────────────────────────────────────────
# 15. Модель 5 — CatBoost (ручной подбор + early stopping)
# ──────────────────────────────────────────────────────────────────────────────
# CatBoost работает с категориальными признаками нативно: вместо OHE применяет
# target encoding с permutation-защитой от утечки (ordered boosting).
# TF-IDF после L1-отбора передаём отдельными числовыми колонками.
print("\n" + "=" * 70)
print("CatBoost")
print("=" * 70)
t_start = time.time()

cb_candidates = [
    {"learning_rate": 0.05, "depth": 6},
    {"learning_rate": 0.1,  "depth": 6},
    {"learning_rate": 0.05, "depth": 8},
]
best_cb_model = None
best_cb_score = -1
best_cb_params = None

for params in cb_candidates:
    candidate = CatBoostClassifier(
        iterations=500, l2_leaf_reg=3,
        loss_function="MultiClass", eval_metric="Accuracy",
        early_stopping_rounds=50, random_seed=42, verbose=False,
        cat_features=CATEGORICAL_FEATURES,
        **params,
    )
    candidate.fit(train_cb, y_train, eval_set=(test_cb, y_test), verbose=False)
    preds = candidate.predict(test_cb).flatten().astype(int)
    score = f1_score(y_test, preds, average="macro")
    print(f"  {params} -> f1_macro={score:.4f}")
    if score > best_cb_score:
        best_cb_score = score
        best_cb_model = candidate
        best_cb_params = params

print(f"Лучшие параметры: {best_cb_params} (f1_macro={best_cb_score:.4f})")
cb_preds = best_cb_model.predict(test_cb).flatten().astype(int)
results["CatBoost"] = evaluate("CatBoost", y_test, cb_preds, time.time() - t_start)

cb_feature_names = NUMERIC_FEATURES + CATEGORICAL_FEATURES + tfidf_col_names
cb_top = pd.Series(best_cb_model.get_feature_importance(), index=cb_feature_names).nlargest(10)
print("\nTop-10 признаков (CatBoost):")
print(cb_top.to_string())


# ──────────────────────────────────────────────────────────────────────────────
# 16. Итоговое сравнение
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Итоговое сравнение моделей")
print("=" * 70)
summary = pd.DataFrame([
    {"Model": name, **metrics} for name, metrics in results.items()
])
print(summary.to_string(index=False))
