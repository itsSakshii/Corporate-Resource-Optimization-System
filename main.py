
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE


BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "Extended_Employee_Performance_and_Productivity_Data.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "Decision_Tree_model.pkl")
PROCESSED_DIR = os.path.join(BASE_DIR, "processed_data")

RANDOM_STATE = 42
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "models"), exist_ok=True)


print("📂 Loading dataset...")
df = pd.read_csv(DATA_PATH)
print(f"✅ Dataset loaded successfully: {df.shape}")


print("\n🧹 Cleaning data...")


df_cleaned = df.drop(columns=[
    'Employee_ID', 'Hire_Date', 'Job_Title'
], errors='ignore')


df_cleaned = df_cleaned.fillna(df_cleaned.mean(numeric_only=True))

print(f"✅ Data cleaned: {df_cleaned.shape}")
print("Columns:", df_cleaned.columns.tolist())


if 'Work_Hours_Per_Week' in df_cleaned.columns and 'Monthly_Salary' in df_cleaned.columns:
    df_cleaned['Efficiency'] = df_cleaned['Monthly_Salary'] / (df_cleaned['Work_Hours_Per_Week'] + 1)


categorical_cols = ['Department', 'Gender', 'Education_Level', 'Resigned']
for col in categorical_cols:
    if col in df_cleaned.columns:
        df_cleaned[col] = df_cleaned[col].astype('category').cat.codes


target_col = 'Performance_Score'
if target_col not in df_cleaned.columns:
    raise KeyError("❌ 'Performance_Score' column missing in dataset!")

X = df_cleaned.drop(columns=[target_col])
y = df_cleaned[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.15, random_state=RANDOM_STATE, stratify=y
)

print("\n✅ Data split complete.")
print("Training data:", X_train.shape, "Test data:", X_test.shape)


print("\n⚖️ Applying SMOTE to balance classes...")
smote = SMOTE(random_state=RANDOM_STATE)
X_train_bal, y_train_bal = smote.fit_resample(X_train, y_train)

print("Before SMOTE:", y_train.value_counts().to_dict())
print("After SMOTE:", y_train_bal.value_counts().to_dict())


print("\n📏 Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_bal)
X_test_scaled = scaler.transform(X_test)


print("\n🚀 Training RandomForestClassifier...")
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=4,
    min_samples_leaf=2,
    random_state=RANDOM_STATE,
    n_jobs=-1
)
model.fit(X_train_scaled, y_train_bal)


y_pred = model.predict(X_test_scaled)

accuracy = accuracy_score(y_test, y_pred)
print(f"\n✅ Model Accuracy: {accuracy * 100:.2f}%\n")
print("Classification Report:\n", classification_report(y_test, y_pred))


print("\n💾 Saving model, scaler, and columns...")

joblib.dump(model, MODEL_PATH)
joblib.dump(scaler, os.path.join(PROCESSED_DIR, "scaler.joblib"))
joblib.dump(X.columns.tolist(), os.path.join(PROCESSED_DIR, "columns.joblib"))


np.save(os.path.join(PROCESSED_DIR, "X_train_scaled.npy"), X_train_scaled)
np.save(os.path.join(PROCESSED_DIR, "X_test_scaled.npy"), X_test_scaled)
y_train.to_csv(os.path.join(PROCESSED_DIR, "y_train.csv"), index=False)
y_test.to_csv(os.path.join(PROCESSED_DIR, "y_test.csv"), index=False)

print("\n✅ Model, Scaler, and Columns saved successfully!")
print(f"📁 Model Path: {MODEL_PATH}")
print(f"📁 Scaler Path: {os.path.join(PROCESSED_DIR, 'scaler.joblib')}")
print(f"📁 Columns Path: {os.path.join(PROCESSED_DIR, 'columns.joblib')}")

print("\n🎯 Training pipeline complete.")
