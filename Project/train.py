# =========================================================
# IMPORT LIBRARY
# =========================================================

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report

from imblearn.over_sampling import SMOTE

# =========================================================
# LOAD DATASET
# =========================================================

df = pd.read_csv(r"C:\Users\Michael Marcel\OneDrive\Desktop\TA\Data\Dataset\student_lifestyle_100k.csv")

# =========================================================
# MENAMPILKAN INFORMASI DATASET
# =========================================================

print("=== DATASET ===")
print(df.head())

print("\n=== INFO DATASET ===")
print(df.info())

print("\n=== DISTRIBUSI TARGET ===")
print(df['Depression'].value_counts())

# =========================================================
# MEMISAHKAN FITUR DAN TARGET
# =========================================================

X = df.drop('Depression', axis=1)
y = df['Depression']

# =========================================================
# ENCODING DATA KATEGORIKAL
# =========================================================

label_encoder = LabelEncoder()

for col in X.select_dtypes(include='object').columns:
    X[col] = label_encoder.fit_transform(X[col])

# =========================================================
# SPLIT DATA TRAINING DAN TESTING
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# =========================================================
# TANPA SMOTE
# =========================================================

# -------------------------
# Logistic Regression
# -------------------------

lr_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

lr_model.fit(X_train, y_train)

y_pred_lr = lr_model.predict(X_test)

# -------------------------
# Decision Tree
# -------------------------

dt_model = DecisionTreeClassifier(
    random_state=42
)

dt_model.fit(X_train, y_train)

y_pred_dt = dt_model.predict(X_test)

# -------------------------
# Random Forest
# -------------------------

rf_model = RandomForestClassifier(
    random_state=42
)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

# =========================================================
# DENGAN SMOTE
# =========================================================

smote = SMOTE(random_state=42)

X_train_smote, y_train_smote = smote.fit_resample(
    X_train,
    y_train
)

print("\n=== DISTRIBUSI SETELAH SMOTE ===")
print(pd.Series(y_train_smote).value_counts())

# -------------------------
# Logistic Regression SMOTE
# -------------------------

lr_model_smote = LogisticRegression(
    max_iter=1000,
    random_state=42
)

lr_model_smote.fit(
    X_train_smote,
    y_train_smote
)

y_pred_lr_smote = lr_model_smote.predict(X_test)

# -------------------------
# Decision Tree SMOTE
# -------------------------

dt_model_smote = DecisionTreeClassifier(
    random_state=42
)

dt_model_smote.fit(
    X_train_smote,
    y_train_smote
)

y_pred_dt_smote = dt_model_smote.predict(X_test)

# -------------------------
# Random Forest SMOTE
# -------------------------

rf_model_smote = RandomForestClassifier(
    random_state=42
)

rf_model_smote.fit(
    X_train_smote,
    y_train_smote
)

y_pred_rf_smote = rf_model_smote.predict(X_test)

# =========================================================
# FUNGSI MENGAMBIL METRIK
# =========================================================

def extract_metrics(y_true, y_pred):

    report = classification_report(
        y_true,
        y_pred,
        output_dict=True,
        zero_division=0
    )

    return {
        'Accuracy': round(report['accuracy'], 2),

        'Precision Non-Depresi':
            round(report['False']['precision'], 2),

        'Recall Non-Depresi':
            round(report['False']['recall'], 2),

        'F1-Score Non-Depresi':
            round(report['False']['f1-score'], 2),

        'Precision Depresi':
            round(report['True']['precision'], 2),

        'Recall Depresi':
            round(report['True']['recall'], 2),

        'F1-Score Depresi':
            round(report['True']['f1-score'], 2)
    }
# =========================================================
# MENGAMBIL HASIL EVALUASI
# =========================================================

# Logistic Regression
lr_no_smote = extract_metrics(y_test, y_pred_lr)
lr_smote = extract_metrics(y_test, y_pred_lr_smote)

# Decision Tree
dt_no_smote = extract_metrics(y_test, y_pred_dt)
dt_smote = extract_metrics(y_test, y_pred_dt_smote)

# Random Forest
rf_no_smote = extract_metrics(y_test, y_pred_rf)
rf_smote = extract_metrics(y_test, y_pred_rf_smote)

# =========================================================
# MEMBUAT TABEL LOGISTIC REGRESSION
# =========================================================

logistic_table = pd.DataFrame([
    ['Tanpa SMOTE', *lr_no_smote.values()],
    ['Dengan SMOTE', *lr_smote.values()]
],
columns=[
    'Metode',
    'Accuracy',
    'Precision Non-Depresi',
    'Recall Non-Depresi',
    'F1-Score Non-Depresi',
    'Precision Depresi',
    'Recall Depresi',
    'F1-Score Depresi'
])

print("\n=== HASIL LOGISTIC REGRESSION ===")
print(logistic_table)

# =========================================================
# MEMBUAT TABEL DECISION TREE
# =========================================================

decision_tree_table = pd.DataFrame([
    ['Tanpa SMOTE', *dt_no_smote.values()],
    ['Dengan SMOTE', *dt_smote.values()]
],
columns=[
    'Metode',
    'Accuracy',
    'Precision Non-Depresi',
    'Recall Non-Depresi',
    'F1-Score Non-Depresi',
    'Precision Depresi',
    'Recall Depresi',
    'F1-Score Depresi'
])

print("\n=== HASIL DECISION TREE ===")
print(decision_tree_table)

# =========================================================
# MEMBUAT TABEL RANDOM FOREST
# =========================================================

random_forest_table = pd.DataFrame([
    ['Tanpa SMOTE', *rf_no_smote.values()],
    ['Dengan SMOTE', *rf_smote.values()]
],
columns=[
    'Metode',
    'Accuracy',
    'Precision Non-Depresi',
    'Recall Non-Depresi',
    'F1-Score Non-Depresi',
    'Precision Depresi',
    'Recall Depresi',
    'F1-Score Depresi'
])

print("\n=== HASIL RANDOM FOREST ===")
print(random_forest_table)

# =========================================================
# MENYIMPAN HASIL KE CSV
# =========================================================

logistic_table.to_csv(
    'hasil_logistic_regression.csv',
    index=False
)

decision_tree_table.to_csv(
    'hasil_decision_tree.csv',
    index=False
)

random_forest_table.to_csv(
    'hasil_random_forest.csv',
    index=False
)

print("\nFile evaluasi berhasil disimpan.")