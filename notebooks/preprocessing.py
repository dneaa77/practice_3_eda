
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# 1. Загружаем датасет
df = pd.read_csv("titanic.csv")

print("Размер датасета:", df.shape)
print("\nПервые 5 строк:")
print(df.head())

# 2. Выбираем признаки и целевую переменную
numeric_features = ["Age", "Fare", "SibSp", "Parch"]
categorical_features = ["Pclass", "Sex", "Embarked"]

X = df[numeric_features + categorical_features]
y = df["Survived"]

print("\nПропуски до обработки:")
print(X.isnull().sum())

# 3. Делим данные на train и test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nРазмер X_train:", X_train.shape)
print("Размер X_test:", X_test.shape)


# 4. Обрабатываем числовые признаки

# Возраст и стоимость билета:
# медиана устойчива к выбросам
median_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Число братьев, сестёр, родителей и детей:
# среднее используем как учебную стратегию заполнения
mean_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

# 5. Обрабатываем категориальные признаки

# Для пола и класса используем наиболее частое значение
mode_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Для порта посадки сохраняем факт отсутствия значения
unknown_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(
        strategy="constant",
        fill_value="Unknown"
    )),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# 6. Объединяем все преобразования
preprocessor = ColumnTransformer(transformers=[
    ("median", median_transformer, ["Age", "Fare"]),
    ("mean", mean_transformer, ["SibSp", "Parch"]),
    ("mode", mode_transformer, ["Pclass", "Sex"]),
    ("unknown", unknown_transformer, ["Embarked"])
])


# 7. Обучаем предобработку только на train
X_train_processed = preprocessor.fit_transform(X_train)

# 8. Применяем готовую предобработку к test
X_test_processed = preprocessor.transform(X_test)

# 9. Проверяем результат
print("\nРазмер после обработки:")
print("Train:", X_train_processed.shape)
print("Test:", X_test_processed.shape)

print("\nКоличество пропусков в train после обработки:",
      pd.isna(X_train_processed).sum())

print("Количество пропусков в test после обработки:",
      pd.isna(X_test_processed).sum())

print("\nМедианы числовых признаков, найденные на train:")
print(
    preprocessor.named_transformers_["num"]
    .named_steps["imputer"].statistics_
)

print("\nПрактика №3 завершена!")

import scipy.sparse as sp

train_check = (
    X_train_processed.toarray()
    if sp.issparse(X_train_processed)
    else X_train_processed
)

test_check = (
    X_test_processed.toarray()
    if sp.issparse(X_test_processed)
    else X_test_processed
)

print(
    "\nПропусков в train после обработки:",
    pd.isna(train_check).sum()
)

print(
    "Пропусков в test после обработки:",
    pd.isna(test_check).sum()
)
X_train_processed = preprocessor.fit_transform(X_train)

print("\nПроверка размерностей:")
print("Train:", X_train_processed.shape)
print("Test:", X_test_processed.shape)

assert X_train_processed.shape[1] == X_test_processed.shape[1]
assert pd.isna(train_check).sum() == 0
assert pd.isna(test_check).sum() == 0

print("Проверки пройдены!")

