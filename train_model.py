import pandas as pd
import numpy as np
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from xgboost import XGBRegressor

# Chargement des données
df = pd.read_csv(r"data\2016_Building_Energy_Benchmarking.csv", encoding='latin1')

# Nettoyage minimal et Feature Engineering
df = df[df['SiteEnergyUse(kBtu)'] > 0]
df['BuildingAge'] = 2016 - df['YearBuilt']

# Sélection des features (basé sur vos notebooks)
features = ['PrimaryPropertyType', 'Neighborhood', 'PropertyGFATotal', 
            'NumberofFloors', 'BuildingAge', 'NumberofBuildings']
X = df[features]
y = np.log1p(df['SiteEnergyUse(kBtu)'])

# Définition du Pipeline de pré-traitement
numeric_features = ['PropertyGFATotal', 'NumberofFloors', 'BuildingAge', 'NumberofBuildings']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_features = ['PrimaryPropertyType', 'Neighborhood']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Pipeline complet avec XGBoost (paramètres optimisés)
model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(n_estimators=500, learning_rate=0.05, max_depth=6))
])

# Entraînement et sauvegarde
model_pipeline.fit(X, y)
joblib.dump(model_pipeline, 'model_pipeline_seattle.joblib')
print("Modèle sauvegardé sous 'model_pipeline_seattle.joblib'")