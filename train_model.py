# -*- coding: utf-8 -*-
import os
import pandas as pd
import numpy as np
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from xgboost import XGBRegressor

# ── Chargement des données ──────────────────────────────────────────────────
DATA_PATH = os.path.join("data", "2016_Building_Energy_Benchmarking.csv")
df = pd.read_csv(DATA_PATH, encoding="latin1")

# ── Nettoyage & Feature Engineering ────────────────────────────────────────
df = df[df["SiteEnergyUse(kBtu)"] > 0].copy()
df["BuildingAge"] = 2016 - df["YearBuilt"]

# ── Sélection des features ──────────────────────────────────────────────────
features = [
    "PrimaryPropertyType",
    "Neighborhood",
    "PropertyGFATotal",
    "NumberofFloors",
    "BuildingAge",
    "NumberofBuildings",
]
X = df[features]
y = np.log1p(df["SiteEnergyUse(kBtu)"])

# ── Split train / test ──────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── Pipelines de pré-traitement ─────────────────────────────────────────────
numeric_features = ["PropertyGFATotal", "NumberofFloors", "BuildingAge", "NumberofBuildings"]
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_features = ["PrimaryPropertyType", "Neighborhood"]
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features),
])

# ── Pipeline complet avec XGBoost ───────────────────────────────────────────
model_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", XGBRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        random_state=42,          # ✅ reproductibilité
        n_jobs=-1,                # ✅ parallélisation
    )),
])

# ── Entraînement ────────────────────────────────────────────────────────────
model_pipeline.fit(X_train, y_train)

# ── Évaluation sur le jeu de test ───────────────────────────────────────────
y_pred = model_pipeline.predict(X_test)
r2  = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("=" * 45)
print(f"  R²  (test) : {r2:.4f}")
print(f"  MAE (test) : {mae:.4f}  (en log-kBtu)")
print("=" * 45)

# ── Sauvegarde ──────────────────────────────────────────────────────────────
joblib.dump(model_pipeline, "model_pipeline_seattle.joblib")
print("✅ Modèle sauvegardé sous 'model_pipeline_seattle.joblib'")
