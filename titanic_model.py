import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE, RandomOverSampler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from joblib import dump
from sklearn.pipeline import Pipeline

data = pd.read_csv("titanic_dataset.csv")

print(data.head())

# MISSING VALUES

missing_summary = pd.DataFrame({
    "Missing Values": data.isnull().sum().sort_values(ascending=False),
    "Missing Percentage": round(data.isnull().mean() * 100, 2).sort_values(ascending=False)
})
print(missing_summary)

median_age = data["age"].median()
data["age"] = data["age"].fillna(median_age)

# DUPLICATED VALUES

print("Duplicate rows:", data.duplicated().sum())
data.drop_duplicates(inplace=True)

# STRING INCONSISTANCY

data.drop(columns=["deck"], inplace=True)

categorical_columns = ["embarked", "embark_town"]
for i in categorical_columns:
    mode = data[i].mode()[0]
    data[i] = data[i].fillna(mode)

missing_summary = pd.DataFrame({
    "Missing Values": data.isnull().sum().sort_values(ascending=False),
    "Missing Percentage": round(data.isnull().mean() * 100, 2).sort_values(ascending=False)
})
print(missing_summary)

data.drop(columns=["embarked", "class", "alive"], inplace=True)

# TRAIN TEST SPLIT

X = data.drop(columns=["survived"])
y = data["survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=1,
    shuffle=True,
    stratify=y
)

print("Train and test shapes:", X_train.shape, X_test.shape)

# DATA PREPROCESSING

# LABEL ENCODING

nominal = ["embark_town", "who"]
binary_ordinal = ["adult_male", "sex", "alone"]

for i in binary_ordinal:
    le = LabelEncoder()
    le.fit(X_train[i])
    X_train[i] = le.transform(X_train[i])
    X_test[i] = le.transform(X_test[i])

# ONE HOT ENCODING

X_train = pd.get_dummies(X_train, columns=nominal, dtype=int)
X_test = pd.get_dummies(X_test, columns=nominal, dtype=int)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

# HANDLING CLASS IMBALANCE

print("Original class distribution:")
print(y_train.value_counts())

# SMOTE

smote = SMOTE()
X_train_SMOTE, y_train_SMOTE = smote.fit_resample(X_train, y_train)
print("After SMOTE:")
print(y_train_SMOTE.value_counts())


pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression())
])
pipeline.fit(X_train, y_train)

prediction_train = pipeline.predict(X_train)
prediction_test = pipeline.predict(X_test)

print(classification_report(y_train, prediction_train))
print(classification_report(y_test, prediction_test))

print(confusion_matrix(y_test, prediction_test))
