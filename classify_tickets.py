import warnings
warnings.filterwarnings("ignore")

import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score
from scipy.sparse import hstack, csr_matrix
import lightgbm as lgb
from catboost import CatBoostClassifier


df = pd.read_csv("twinby_support_tickets.csv")
print(f"Загружено {len(df):,} тикетов, {df['category'].nunique()} категорий")
print(df["category"].value_counts().to_string())


# Задача — роутинг тикета: в момент поступления обращения нам известны
# только данные о пользователе и текст сообщения. Поля resolution_time_minutes,
# escalated, csat_ticket и support_response появляются уже после того, как агент
# обработал тикет — включать их в модель нельзя
NUMERIC = [
    "hour_of_day", "age", "days_since_registration",
    "total_likes", "total_matches", "messages_sent",
    "active_days_last_week", "boosts_purchased", "superlikes_purchased",
    "past_tickets_count", "avg_csat_history",
    "message_length_chars", "message_length_words", "sentiment_score",
]
CATEGORICAL = ["day_of_week", "gender", "subscription", "platform", "channel", "language", "priority"]
TEXT = "user_message"
TARGET = "category"

X = df[NUMERIC + CATEGORICAL + [TEXT]].copy()
y = df[TARGET]

# avg_csat_history отсутствует у новых пользователей, у которых ещё не было
# закрытых тикетов. Заполняем медианой по всей колонке — нейтральное значение
for col in NUMERIC:
    X[col] = pd.to_numeric(X[col], errors="coerce").fillna(X[col].median())
for col in CATEGORICAL:
    X[col] = X[col].fillna("unknown").astype(str)
X[TEXT] = X[TEXT].fillna("").astype(str)

# LabelEncoder переводит строковые метки в числа в алфавитном порядке:
# account=0, bug=1, complaint=2, moderation=3, other=4, payment=5, refund=6
label_enc = LabelEncoder()
y_enc = label_enc.fit_transform(y)
classes = label_enc.classes_

# stratify=y_enc гарантирует, что доля каждого класса в train и test одинакова.
# Без этого редкие классы (other — 4.5%, refund — 8.2%) могут случайно
# оказаться недопредставлены в тесте и метрики по ним будут ненадёжны
X_train, X_test, y_train, y_test = train_test_split(
    X, y_enc, test_size=0.2, stratify=y_enc, random_state=42
)
X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
print(f"\nTrain: {len(X_train):,} | Test: {len(X_test):,}")


# TF-IDF превращает текст в числовой вектор: каждое слово (или биграмма) —
# отдельный признак, значение = частота слова в документе, делённая на
# частоту во всём корпусе (инверсная). Биграммы (ngram_range=(1,2)) нужны
# чтобы захватывать устойчивые словосочетания: "не работает", "вернуть деньги",
# "не могу войти". sublinear_tf=True логарифмирует частоту — слово встреченное
# 10 раз не будет в 10 раз важнее встреченного 1 раз
tfidf = TfidfVectorizer(max_features=500, ngram_range=(1, 2), min_df=3, sublinear_tf=True)
train_tfidf = tfidf.fit_transform(X_train[TEXT])
test_tfidf = tfidf.transform(X_test[TEXT])

# Logistic Regression и Random Forest из sklearn работают со sparse-матрицами
# (разреженными). TF-IDF по природе разреженный — большинство слов в конкретном
# сообщении отсутствуют, то есть значение = 0. Хранить нули в памяти не нужно.
# Числовые признаки оборачиваем в csr_matrix, категориальные кодируем через
# OneHotEncoder (каждое уникальное значение → отдельная колонка с 0 или 1),
# затем всё склеиваем в одну разреженную матрицу через hstack
ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=True)
train_sparse = hstack([
    train_tfidf,
    csr_matrix(X_train[NUMERIC].values.astype(float)),
    ohe.fit_transform(X_train[CATEGORICAL]),
])
test_sparse = hstack([
    test_tfidf,
    csr_matrix(X_test[NUMERIC].values.astype(float)),
    ohe.transform(X_test[CATEGORICAL]),
])

# LightGBM не принимает sparse-матрицы напрямую, поэтому собираем плотный массив.
# Категориальные признаки кодируем через OrdinalEncoder (platform → 0,1,2...),
# но передаём LightGBM индексы этих колонок через categorical_feature —
# тогда модель обрабатывает их как категории, а не как числа
ord_enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=999)
train_cat_enc = ord_enc.fit_transform(X_train[CATEGORICAL]).astype(int)
test_cat_enc = ord_enc.transform(X_test[CATEGORICAL]).astype(int)

train_dense = np.hstack([X_train[NUMERIC].values.astype(float), train_cat_enc, train_tfidf.toarray()])
test_dense = np.hstack([X_test[NUMERIC].values.astype(float), test_cat_enc, test_tfidf.toarray()])
cat_feature_idx = list(range(len(NUMERIC), len(NUMERIC) + len(CATEGORICAL)))

# CatBoost не требует предварительного кодирования категорий совсем —
# он сам строит target encoding внутри: для каждого значения категории
# считает статистику по целевой переменной с хитрым permutation-трюком,
# чтобы не переобучиться. Поэтому передаём сырые строки ("iOS", "Android")
# и просто указываем имена колонок через cat_features
tfidf_col_names = [f"tfidf_{i}" for i in range(train_tfidf.shape[1])]
train_cb = pd.concat([
    X_train[NUMERIC],
    X_train[CATEGORICAL],
    pd.DataFrame(train_tfidf.toarray(), columns=tfidf_col_names),
], axis=1)
test_cb = pd.concat([
    X_test[NUMERIC],
    X_test[CATEGORICAL],
    pd.DataFrame(test_tfidf.toarray(), columns=tfidf_col_names),
], axis=1)


def print_metrics(name, y_true, y_pred, elapsed):
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average="macro")
    f1w = f1_score(y_true, y_pred, average="weighted")
    print(f"\n{name} | {elapsed:.1f}s | acc={acc:.4f} | f1_macro={f1m:.4f} | f1_weighted={f1w:.4f}")
    # F1 macro усредняет метрику по классам без учёта их размера — каждый класс
    # весит одинаково. Это честнее для нашей задачи, потому что payment (25%)
    # не должен "перекрывать" ошибки на other (4.5%). F1 weighted, наоборот,
    # взвешивает по количеству примеров — удобен для общей картины
    print(classification_report(y_true, y_pred, target_names=classes, digits=4, zero_division=0))
    return {
        "accuracy": float(acc),
        "f1_macro": float(f1m),
        "f1_weighted": float(f1w),
        "train_time": round(elapsed, 1),
    }


results = {}

# Logistic Regression строит линейную границу между классами в пространстве
# признаков. Параметр C=1.0 — сила регуляризации (чем меньше C, тем сильнее
# штраф за большие веса, тем проще модель). solver="lbfgs" — алгоритм
# оптимизации, хорошо работает на многоклассовых задачах
print("\n--- Logistic Regression ---")
t0 = time.time()
lr = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs", random_state=42, n_jobs=-1)
lr.fit(train_sparse, y_train)
results["Logistic Regression"] = print_metrics(
    "Logistic Regression", y_test, lr.predict(test_sparse), time.time() - t0
)

# Random Forest строит 300 независимых деревьев решений на случайных подвыборках
# данных и признаков, затем усредняет предсказания. Не нужно масштабировать
# признаки (деревья сравнивают пороги, а не величины). min_samples_leaf=2
# не позволяет листьям содержать единственный пример — небольшая защита от переобучения
print("\n--- Random Forest ---")
t0 = time.time()
rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1)
rf.fit(train_sparse, y_train)
results["Random Forest"] = print_metrics(
    "Random Forest", y_test, rf.predict(test_sparse), time.time() - t0
)

# feature_importances_ показывает, насколько каждый признак снижает
# неопределённость (impurity) при разбиениях в деревьях — по сути,
# вклад признака в качество классификации
rf_feature_names = list(tfidf.get_feature_names_out()) + NUMERIC + list(ohe.get_feature_names_out(CATEGORICAL))
rf_importance = pd.Series(rf.feature_importances_, index=rf_feature_names).nlargest(10)
print("Top-10 features (RF):\n", rf_importance.to_string())

# LightGBM — градиентный бустинг: каждое следующее дерево обучается на ошибках
# предыдущих. learning_rate=0.05 — маленький шаг, чтобы не перепрыгнуть минимум.
# num_leaves=63 — сложность каждого дерева. subsample и colsample_bytree — доля
# данных и признаков для каждого дерева (аналог bagging, снижает переобучение).
# early_stopping: если за 50 итераций качество на тестовой выборке не растёт —
# останавливаемся, чтобы не переобучиться на обучающей
print("\n--- LightGBM ---")
t0 = time.time()
lgbm = lgb.LGBMClassifier(
    n_estimators=500, learning_rate=0.05, num_leaves=63,
    subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
    random_state=42, n_jobs=-1, verbose=-1,
)
lgbm.fit(
    train_dense, y_train,
    categorical_feature=cat_feature_idx,
    eval_set=[(test_dense, y_test)],
    callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=0)],
)
results["LightGBM"] = print_metrics(
    "LightGBM", y_test, lgbm.predict(test_dense), time.time() - t0
)

lgb_feature_names = NUMERIC + CATEGORICAL + list(tfidf.get_feature_names_out())
lgb_importance = pd.Series(lgbm.feature_importances_, index=lgb_feature_names).nlargest(10)
print("Top-10 features (LightGBM):\n", lgb_importance.to_string())

# CatBoost — тоже градиентный бустинг, но с двумя ключевыми отличиями:
# 1) ordered boosting: деревья строятся с учётом порядка объектов, что снижает
#    смещение оценок на обучающей выборке
# 2) нативный target encoding для категорий: вместо OrdinalEncoder модель сама
#    превращает "iOS"/"Android" в числа, считая статистики по целевой переменной
#    с permutation-защитой от переобучения
# l2_leaf_reg=3 — L2-регуляризация листьев дерева
print("\n--- CatBoost ---")
t0 = time.time()
catboost = CatBoostClassifier(
    iterations=500, learning_rate=0.05, depth=6, l2_leaf_reg=3,
    loss_function="MultiClass", eval_metric="Accuracy",
    early_stopping_rounds=50, random_seed=42, verbose=False,
    cat_features=CATEGORICAL,
)
catboost.fit(train_cb, y_train, eval_set=(test_cb, y_test), verbose=False)
cb_preds = catboost.predict(test_cb).flatten().astype(int)
results["CatBoost"] = print_metrics(
    "CatBoost", y_test, cb_preds, time.time() - t0
)

cb_feature_names = NUMERIC + CATEGORICAL + tfidf_col_names
cb_importance = pd.Series(catboost.get_feature_importance(), index=cb_feature_names).nlargest(10)
print("Top-10 features (CatBoost):\n", cb_importance.to_string())


print("\nИтоговое сравнение:")
summary = pd.DataFrame([
    {"Model": name, "Accuracy": d["accuracy"], "F1 macro": d["f1_macro"],
     "F1 weighted": d["f1_weighted"], "Time (s)": d["train_time"]}
    for name, d in results.items()
])
print(summary.to_string(index=False))
