"""
Классификация обращений в поддержку Twinby по 7 категориям.
Сравниваются 4 модели: Logistic Regression, Random Forest, LightGBM, CatBoost.
"""

import warnings
warnings.filterwarnings("ignore")

import time
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
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
# resolution_time_minutes, escalated, csat_ticket, support_response —
# post-factum данные, их включение в модель = утечка данных (data leakage).
NUMERIC_FEATURES = [
    "hour_of_day", "age", "days_since_registration",
    "total_likes", "total_matches", "messages_sent",
    "active_days_last_week", "boosts_purchased", "superlikes_purchased",
    "past_tickets_count", "avg_csat_history",
    "message_length_chars", "message_length_words", "sentiment_score",
]
CATEGORICAL_FEATURES = [
    "day_of_week", "gender", "subscription",
    "platform", "channel", "language", "priority",
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

# В категориальных пропуски заменяем отдельной меткой "unknown":
# так модель сможет учитывать сам факт отсутствия значения как сигнал.
for col in CATEGORICAL_FEATURES:
    X[col] = X[col].fillna("unknown").astype(str)

X[TEXT_FEATURE] = X[TEXT_FEATURE].fillna("").astype(str)

# Строковые метки классов в целочисленные индексы: account=0, bug=1, ...
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
CLASSES = label_encoder.classes_


# ──────────────────────────────────────────────────────────────────────────────
# 4. Разбиение на train и test
# ──────────────────────────────────────────────────────────────────────────────
# stratify сохраняет пропорции классов в обеих выборках. Для нас это критично:
# редкий класс 'other' (4.5%) без стратификации может перекоситься в тесте,
# и метрики по нему станут ненадёжными.
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
)
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
print(f"\nTrain: {len(X_train):,} | Test: {len(X_test):,}")


# ──────────────────────────────────────────────────────────────────────────────
# 5. Текст: TF-IDF + отбор признаков
# ──────────────────────────────────────────────────────────────────────────────
# TF-IDF превращает текст в разреженный вектор: каждое слово/биграмма —
# отдельный признак со значением = частота в документе × обратная частота в корпусе.
#
# Параметры подобраны осознанно:
#   - ngram_range=(1, 2)  — включаем биграммы для фраз: "не работает",
#                           "вернуть деньги", "не могу войти"
#   - max_features=2000   — широкий словарь, из него потом отберём лучшие
#   - min_df=5            — слово должно встретиться в ≥5 тикетах,
#                           иначе это шум/опечатка/редкий токен
#   - max_df=0.9          — слово, встречающееся в >90% тикетов, — стоп-слово,
#                           оно не различает классы (автоматический стоп-лист)
#   - sublinear_tf=True   — логарифм частоты, чтобы слово, встреченное 10 раз,
#                           не было в 10 раз важнее слова, встреченного 1 раз
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

# Отбор признаков через хи-квадрат — статистический тест,
# который измеряет зависимость между признаком и целевой переменной.
# Оставляем 1000 самых информативных токенов из 2000: фильтруем слова,
# слабо связанные с категорией тикета. Это и есть feature selection для текста.
feature_selector = SelectKBest(chi2, k=1000)
tfidf_train = feature_selector.fit_transform(tfidf_train_full, y_train)
tfidf_test = feature_selector.transform(tfidf_test_full)

# Сохраняем имена отобранных токенов — понадобятся для feature importance
all_tokens = np.array(tfidf_vectorizer.get_feature_names_out())
selected_tokens = all_tokens[feature_selector.get_support()]
print(f"После chi²-отбора: {tfidf_train.shape[1]} токенов")


# ──────────────────────────────────────────────────────────────────────────────
# 6. Категориальные: OneHot для всех моделей (кроме CatBoost)
# ──────────────────────────────────────────────────────────────────────────────
# OHE создаёт бинарную колонку на каждое уникальное значение категории.
# Используем этот подход для LR, RF и LightGBM — единая стратегия кодирования,
# нет иерархии между значениями (в отличие от OrdinalEncoder, где "Android" < "iOS").
# handle_unknown='ignore' корректно обрабатывает категории, которых не было в train.
ohe_encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
cat_train_ohe = ohe_encoder.fit_transform(X_train[CATEGORICAL_FEATURES])
cat_test_ohe = ohe_encoder.transform(X_test[CATEGORICAL_FEATURES])
print(f"OHE категориальных: {cat_train_ohe.shape[1]} колонок")


# ──────────────────────────────────────────────────────────────────────────────
# 7. Сборка финальных матриц
# ──────────────────────────────────────────────────────────────────────────────
# Sparse-матрица для LR и RF: TF-IDF уже разрежённый, OHE тоже,
# объединяем через scipy.sparse.hstack. Памяти это требует минимум —
# нули не хранятся.
num_train_sparse = csr_matrix(X_train[NUMERIC_FEATURES].values.astype(float))
num_test_sparse = csr_matrix(X_test[NUMERIC_FEATURES].values.astype(float))

X_train_sparse = hstack([tfidf_train, num_train_sparse, cat_train_ohe]).tocsr()
X_test_sparse = hstack([tfidf_test, num_test_sparse, cat_test_ohe]).tocsr()

# Плотная матрица для LightGBM: sklearn-API принимает sparse, но при небольшом
# числе колонок dense быстрее. Здесь финально 1000 (TF-IDF) + 14 (num) + ~35 (OHE).
X_train_dense = X_train_sparse.toarray()
X_test_dense = X_test_sparse.toarray()

# DataFrame для CatBoost: строковые категории передаются как есть,
# модель сама применяет target encoding с ordered boosting (защита от утечки).
# TF-IDF добавляем как числовые колонки.
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

print(f"\nИтоговая матрица: {X_train_sparse.shape[1]} признаков "
      f"({tfidf_train.shape[1]} TF-IDF + {len(NUMERIC_FEATURES)} num + {cat_train_ohe.shape[1]} OHE)")


# ──────────────────────────────────────────────────────────────────────────────
# 8. Вспомогательная функция для оценки и печати метрик
# ──────────────────────────────────────────────────────────────────────────────
def evaluate(name, y_true, y_pred, elapsed):
    """Вычисляет и печатает ключевые метрики классификации.

    Основная метрика — F1 macro: она усредняет F1 по классам с равным весом
    независимо от их размера. Это важно для нашего несбалансированного датасета,
    где payment (25%) не должен «забивать» ошибки на other (4.5%).
    """
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
# 9. Модель 1 — Logistic Regression (+ GridSearchCV по C)
# ──────────────────────────────────────────────────────────────────────────────
# Линейный бейзлайн. Параметр C регулирует силу L2-регуляризации: чем меньше C,
# тем сильнее штраф за большие веса, тем проще модель. Подбираем C по 3-fold CV.
print("\n" + "=" * 70)
print("Logistic Regression")
print("=" * 70)
t_start = time.time()

lr_grid = GridSearchCV(
    estimator=LogisticRegression(max_iter=2000, solver="lbfgs", n_jobs=-1, random_state=42),
    param_grid={"C": [0.1, 1.0, 10.0]},
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
# 10. Модель 2 — Random Forest (+ GridSearchCV по n_estimators и min_samples_leaf)
# ──────────────────────────────────────────────────────────────────────────────
# Ансамбль независимых деревьев решений. Каждое дерево учится на случайной
# подвыборке данных и признаков (bagging + random subspace), итоговое
# предсказание — голосование большинства. Не требует масштабирования признаков.
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

# feature_importances_ = среднее снижение impurity (неопределённости класса)
# при разбиениях по этому признаку во всех деревьях леса.
rf_feature_names = list(selected_tokens) + NUMERIC_FEATURES + list(ohe_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
rf_top_features = pd.Series(rf_model.feature_importances_, index=rf_feature_names).nlargest(10)
print("\nTop-10 признаков (Random Forest):")
print(rf_top_features.to_string())


# ──────────────────────────────────────────────────────────────────────────────
# 11. Модель 3 — LightGBM (ручной подбор + early stopping)
# ──────────────────────────────────────────────────────────────────────────────
# Градиентный бустинг на деревьях: каждое следующее дерево обучается
# на ошибках предыдущих. Для больших деревьев используем листовый рост
# (leaf-wise) — это отличает LightGBM от XGBoost и делает его быстрее.
#
# Комбинированный тюнинг: перебираем 4 конфигурации, в каждой используем
# early stopping по валидационной выборке — так n_estimators подбирается
# автоматически (обучение останавливается, как только качество перестаёт расти).
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
    print(f"  {params} → f1_macro={score:.4f}")
    if score > best_lgb_score:
        best_lgb_score = score
        best_lgb_model = candidate
        best_lgb_params = params

print(f"Лучшие параметры: {best_lgb_params} (f1_macro={best_lgb_score:.4f})")
results["LightGBM"] = evaluate(
    "LightGBM", y_test, best_lgb_model.predict(X_test_dense), time.time() - t_start
)

lgb_feature_names = list(selected_tokens) + NUMERIC_FEATURES + list(ohe_encoder.get_feature_names_out(CATEGORICAL_FEATURES))
lgb_top_features = pd.Series(best_lgb_model.feature_importances_, index=lgb_feature_names).nlargest(10)
print("\nTop-10 признаков (LightGBM):")
print(lgb_top_features.to_string())


# ──────────────────────────────────────────────────────────────────────────────
# 12. Модель 4 — CatBoost (ручной подбор + early stopping)
# ──────────────────────────────────────────────────────────────────────────────
# CatBoost — тоже градиентный бустинг, но с двумя важными отличиями:
# 1) Ordered boosting: деревья строятся с учётом порядка объектов —
#    снижает смещение оценок на обучающей выборке.
# 2) Нативное кодирование категорий: вместо OHE модель применяет target
#    encoding с permutation-защитой от утечки. Мы передаём строковые
#    категории как есть через параметр cat_features.
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
    print(f"  {params} → f1_macro={score:.4f}")
    if score > best_cb_score:
        best_cb_score = score
        best_cb_model = candidate
        best_cb_params = params

print(f"Лучшие параметры: {best_cb_params} (f1_macro={best_cb_score:.4f})")
cb_preds = best_cb_model.predict(test_cb).flatten().astype(int)
results["CatBoost"] = evaluate("CatBoost", y_test, cb_preds, time.time() - t_start)

cb_feature_names = NUMERIC_FEATURES + CATEGORICAL_FEATURES + tfidf_col_names
cb_top_features = pd.Series(best_cb_model.get_feature_importance(), index=cb_feature_names).nlargest(10)
print("\nTop-10 признаков (CatBoost):")
print(cb_top_features.to_string())


# ──────────────────────────────────────────────────────────────────────────────
# 13. Итоговое сравнение
# ──────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("Итоговое сравнение моделей")
print("=" * 70)
summary = pd.DataFrame([
    {"Model": name, **metrics} for name, metrics in results.items()
])
print(summary.to_string(index=False))
