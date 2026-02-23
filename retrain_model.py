"""Retrain Random Forest on UNSCALED data to fix overfitting.

Root cause: X_train.csv had standardized area values (mean=0, std=1),
but the app sends raw area values. This script retrains on the original
unscaled Housing_cleaned.csv so the model works directly with raw inputs.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load original unscaled data
df = pd.read_csv('data/cleaned/Housing_cleaned.csv')
print(f"Dataset: {df.shape[0]} rows, {df.shape[1]} columns")

X = df.drop('price', axis=1)
y = df['price']

print(f"Features: {list(X.columns)}")
print(f"Area range: {X['area'].min()} - {X['area'].max()}")
print(f"Price range: ₹{y.min():,.0f} - ₹{y.max():,.0f}\n")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows\n")

# Constrained RF — prevents overfitting on 500-row dataset
best_rf = RandomForestRegressor(
    n_estimators=200,
    max_depth=8,
    min_samples_leaf=5,
    min_samples_split=8,
    max_features='sqrt',
    random_state=42
)

best_rf.fit(X_train, y_train)
preds = best_rf.predict(X_test)

print(f"MAE  : ₹{mean_absolute_error(y_test, preds):,.0f}")
print(f"RMSE : ₹{np.sqrt(mean_squared_error(y_test, preds)):,.0f}")
print(f"R²   : {r2_score(y_test, preds):.4f}")

# Feature importances
print("\nFeature Importances:")
for feat, imp in sorted(zip(X_train.columns, best_rf.feature_importances_), key=lambda x: -x[1]):
    print(f"  {feat:20s}: {imp:.4f} ({imp*100:.1f}%)")

# Sanity check — all three prices MUST be different
print("\nSanity Check (area variation):")
prices = []
for area in [500, 3000, 8000]:
    t = pd.DataFrame([{
        'area': area, 'bedrooms': 3, 'bathrooms': 2, 'stories': 2,
        'mainroad': 1, 'guestroom': 0, 'basement': 0,
        'hotwaterheating': 0, 'airconditioning': 1,
        'parking': 1, 'prefarea': 0, 'furnishingstatus': 1
    }])
    t = t[X_train.columns]
    price = best_rf.predict(t)[0]
    prices.append(price)
    print(f"  area={area:>5} → ₹{price:,.0f}")

unique_prices = len(set(f"{p:.0f}" for p in prices))
if unique_prices == 3:
    print(f"\n✅ SUCCESS: All 3 prices are different — model is working correctly!")
    joblib.dump(best_rf, 'models/best_rf_model.joblib')
    print("Model saved to models/best_rf_model.joblib")
else:
    print(f"\n⚠️ Only {unique_prices}/3 unique prices")
