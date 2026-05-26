# student_depression_pipeline_balanced.py

import pandas as pd
import pickle
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

from imblearn.over_sampling import SMOTE

# 1. Load Dataset
df = pd.read_csv(r"C:\Users\Michael Marcel\OneDrive\Desktop\TA\Data\Dataset\student_lifestyle_100k.csv")

# 2. Preprocessing
df = df.drop('Student_ID', axis=1)
df = pd.get_dummies(df, columns=['Gender', 'Department'])

X = df.drop('Depression', axis=1)
y = df['Depression']

# 3. Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 4. Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 5. Handle Imbalance with SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

# 6. Model with class_weight
rf_model = RandomForestClassifier(class_weight='balanced', random_state=42)
rf_model.fit(X_resampled, y_resampled)

# 7. Evaluation
y_pred = rf_model.predict(X_test)

print("\n=== Random Forest (Balanced + SMOTE) ===")
print(classification_report(y_test, y_pred))

# 8. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix - Balanced Random Forest")
plt.show()

# 9. Feature Importance
feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': rf_model.feature_importances_
}).sort_values(by='importance', ascending=False)

print("\nFeature Importance:")
print(feature_importance)

# 10. Save Model
pickle.dump(rf_model, open('model_depression_balanced.pkl', 'wb'))
pickle.dump(scaler, open('scaler_balanced.pkl', 'wb'))

print("\nBalanced model and scaler saved successfully!")
