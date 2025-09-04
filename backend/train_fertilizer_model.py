# import pandas as pd
# import lightgbm as lgb
# import joblib
# import os
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.metrics import accuracy_score

# # 📂 Paths
# DATA_PATH = "data/Fertilizer_recommendation.csv"
# MODEL_PATH = "models/fertilizer_model.pkl"

# # ✅ Load dataset
# print("📥 Loading dataset...")
# df = pd.read_csv(DATA_PATH)

# # 🎯 Target = Fertilizer Name
# target = "Fertilizer Name"

# # 🔹 Features (with optional handling)
# mandatory_features = ["N", "P", "K", "Temparature", "Moisture"]
# optional_features = ["Crop", "Humidity", "Soil Type"]

# features = mandatory_features + optional_features

# X = df[features].copy()
# y = df[target]

# # Encode categorical columns
# label_encoders = {}
# for col in ["Crop", "Soil Type"]:
#     X[col] = X[col].fillna("Unknown")  # if missing, mark as "Unknown"
#     le = LabelEncoder()
#     X[col] = le.fit_transform(X[col])
#     label_encoders[col] = le

# # Fill missing numerical optional fields with mean
# for col in ["Humidity"]:
#     if col in X.columns:
#         X[col] = X[col].fillna(X[col].mean())

# # Encode target
# fertilizer_encoder = LabelEncoder()
# y = fertilizer_encoder.fit_transform(y)

# # Split data
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )

# # ✅ Train LightGBM model
# print("🚀 Training Fertilizer Recommendation Model...")
# model = lgb.LGBMClassifier(n_estimators=200, learning_rate=0.1, random_state=42)
# model.fit(X_train, y_train)

# # ✅ Evaluate
# y_pred = model.predict(X_test)
# acc = accuracy_score(y_test, y_pred)
# print(f"📊 Accuracy: {acc*100:.2f}%")

# # ✅ Save model + encoders + feature info
# os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
# joblib.dump({
#     "model": model,
#     "fertilizer_encoder": fertilizer_encoder,
#     "label_encoders": label_encoders,
#     "mandatory_features": mandatory_features,
#     "optional_features": optional_features
# }, MODEL_PATH)

# print(f"✅ Fertilizer model saved at {MODEL_PATH}")

# _------------------------------------------------------------------------------------Tried Different Approach Same result (14.50%)

# import pandas as pd
# import lightgbm as lgb
# import joblib
# import os
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
# from sklearn.compose import ColumnTransformer
# from sklearn.pipeline import Pipeline
# from sklearn.metrics import accuracy_score

# # 📂 Paths
# DATA_PATH = "data/Fertilizer_recommendation.csv"
# MODEL_PATH = "models/fertilizer_model.pkl"

# # ✅ Load dataset
# print("📥 Loading dataset...")
# df = pd.read_csv(DATA_PATH)

# # 🎯 Target = Fertilizer Name
# target = "Fertilizer Name"

# mandatory_features = ["N", "P", "K", "Temparature", "Moisture"]
# optional_features = ["Crop", "Humidity", "Soil Type"]
# features = mandatory_features + optional_features

# X = df[features].copy()
# y = df[target]

# # Fill missing optional values
# X["Crop"] = X["Crop"].fillna("Unknown")
# X["Soil Type"] = X["Soil Type"].fillna("Unknown")
# X["Humidity"] = X["Humidity"].fillna(X["Humidity"].mean())

# # Encode target
# fertilizer_encoder = LabelEncoder()
# y = fertilizer_encoder.fit_transform(y)

# # Split
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42, stratify=y
# )

# # Column types
# categorical_cols = ["Crop", "Soil Type"]
# numeric_cols = ["N", "P", "K", "Temparature", "Moisture", "Humidity"]

# # 🔹 Preprocessor: scale numerics + one-hot encode categoricals
# preprocessor = ColumnTransformer(
#     transformers=[
#         ("num", StandardScaler(), numeric_cols),
#         ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
#     ]
# )

# # 🔹 Model pipeline
# model = Pipeline(steps=[
#     ("preprocessor", preprocessor),
#     ("classifier", lgb.LGBMClassifier(
#         n_estimators=500,
#         learning_rate=0.05,
#         num_leaves=64,
#         max_depth=-1,
#         random_state=42,
#         class_weight="balanced"
#     ))
# ])

# # 🚀 Train
# print("🚀 Training Fertilizer Recommendation Model...")
# model.fit(X_train, y_train)

# # ✅ Evaluate
# y_pred = model.predict(X_test)
# acc = accuracy_score(y_test, y_pred)
# print(f"📊 Accuracy: {acc*100:.2f}%")

# # ✅ Save
# os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
# joblib.dump({
#     "model": model,
#     "fertilizer_encoder": fertilizer_encoder,
#     "mandatory_features": mandatory_features,
#     "optional_features": optional_features
# }, MODEL_PATH)

# print(f"✅ Fertilizer model saved at {MODEL_PATH}")

# --------------------------------------------------------------Random Forest

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load dataset
df = pd.read_csv("data/Fertilizer_recommendation.csv")

# Features & target
X = df.drop(columns=["Fertilizer Name"])
y = df["Fertilizer Name"]

# Define columns
numeric_features = ["N", "P", "K", "Temparature", "Moisture", "Humidity"]
categorical_features = ["Crop", "Soil Type"]

# Preprocessing
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# Random Forest Classifier
model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42
    ))
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Train
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("✅ Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save model
joblib.dump(model, "models/fertilizer_model.pkl")
print("💾 Fertilizer model saved at models/fertilizer_model.pkl")
