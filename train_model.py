import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from config import (
MODEL_DIR,
OUTPUT_DIR,
MODEL_PATH,
FEATURE_PATH,
RANDOM_STATE,
TEST_SIZE
)

# ============================================

# CREATE REQUIRED DIRECTORIES

# ============================================

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================

# LOAD CALIFORNIA HOUSING DATASET

# ============================================

print("\nLoading California Housing Dataset...\n")

housing = fetch_california_housing()

X = pd.DataFrame(
housing.data,
columns=housing.feature_names
)

y = pd.Series(
housing.target,
name="HousePrice"
)

df = pd.concat([X, y], axis=1)

print("Dataset Shape:", df.shape)

print("\nFirst Five Rows:")
print(df.head())

# ============================================

# DATA QUALITY CHECK

# ============================================

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:")
print(df.duplicated().sum())

print("\nDataset Statistics:")
print(df.describe())

# ============================================

# CORRELATION ANALYSIS

# ============================================

correlation_matrix = df.corr()

target_correlation = correlation_matrix["HousePrice"].sort_values(
ascending=False
)

print("\nCorrelation With Target:")
print(target_correlation)

plt.figure(figsize=(12, 8))

sns.heatmap(
correlation_matrix,
annot=True,
cmap="coolwarm",
fmt=".2f"
)

plt.title("Correlation Matrix")

plt.tight_layout()

plt.savefig(
os.path.join(
OUTPUT_DIR,
"correlation_heatmap.png"
)
)

plt.close()

# ============================================

# TRAIN TEST SPLIT

# ============================================

X_train, X_test, y_train, y_test = train_test_split(
X,
y,
test_size=TEST_SIZE,
random_state=RANDOM_STATE
)

print("\nTraining Samples:", X_train.shape)
print("Testing Samples:", X_test.shape)

# ============================================

# MODEL 1: LINEAR REGRESSION

# ============================================

print("\nTraining Linear Regression...\n")

linear_pipeline = Pipeline([
("scaler", StandardScaler()),
("model", LinearRegression())
])

linear_pipeline.fit(X_train, y_train)

linear_predictions = linear_pipeline.predict(X_test)

# ============================================

# MODEL 2: GRADIENT BOOSTING

# ============================================

print("\nTraining Gradient Boosting...\n")

gradient_model = GradientBoostingRegressor(
random_state=RANDOM_STATE
)

# ============================================

# HYPERPARAMETER TUNING

# ============================================

parameter_grid = {
"n_estimators": [100, 200],
"learning_rate": [0.05, 0.1],
"max_depth": [2, 3],
"min_samples_split": [2, 5]
}

grid_search = GridSearchCV(
estimator=gradient_model,
param_grid=parameter_grid,
cv=5,
scoring="neg_root_mean_squared_error",
n_jobs=-1
)

grid_search.fit(
X_train,
y_train
)

best_gb_model = grid_search.best_estimator_

print("Best Parameters:")
print(grid_search.best_params_)

# ============================================

# CROSS VALIDATION

# ============================================

cv_scores = cross_val_score(
best_gb_model,
X_train,
y_train,
cv=5,
scoring="neg_root_mean_squared_error"
)

cv_rmse = -cv_scores.mean()

print("\nCross Validation RMSE:", cv_rmse)

# ============================================

# FINAL PREDICTIONS

# ============================================

gb_predictions = best_gb_model.predict(X_test)

# ============================================

# MODEL EVALUATION FUNCTION

# ============================================

def evaluate_model(name, actual, predicted):

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predicted
        )
    )

    mae = mean_absolute_error(
        actual,
        predicted
    )

    r2 = r2_score(
        actual,
        predicted
    )

    print("\n" + "=" * 45)
    print(name)
    print("=" * 45)

    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2 Score: {r2:.4f}")

    return {
        "Model": name,
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2
    }

# MODEL EVALUATION



linear_result = evaluate_model(
"Linear Regression",
y_test,
linear_predictions
)

gb_result = evaluate_model(
"Gradient Boosting",
y_test,
gb_predictions
)

# ============================================

# MODEL COMPARISON

# ============================================

results_df = pd.DataFrame([
linear_result,
gb_result
])

print("\nMODEL COMPARISON:")
print(results_df)

results_df.to_csv(
os.path.join(
OUTPUT_DIR,
"model_comparison.csv"
),
index=False
)

# ============================================

# ACTUAL VS PREDICTED GRAPH

# ============================================

plt.figure(figsize=(8, 6))

plt.scatter(
y_test,
gb_predictions,
alpha=0.6
)

plt.xlabel("Actual House Price")
plt.ylabel("Predicted House Price")
plt.title("Actual vs Predicted Values")

minimum = min(
y_test.min(),
gb_predictions.min()
)

maximum = max(
y_test.max(),
gb_predictions.max()
)

plt.plot(
[minimum, maximum],
[minimum, maximum],
"r--"
)

plt.tight_layout()

plt.savefig(
os.path.join(
OUTPUT_DIR,
"actual_vs_predicted.png"
)
)

plt.close()

# ============================================

# RESIDUAL ANALYSIS

# ============================================

residuals = y_test - gb_predictions

plt.figure(figsize=(8, 6))

plt.scatter(
gb_predictions,
residuals,
alpha=0.6
)

plt.axhline(
y=0,
color="red",
linestyle="--"
)

plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residual Plot")

plt.tight_layout()

plt.savefig(
os.path.join(
OUTPUT_DIR,
"residual_plot.png"
)
)

plt.close()

# ============================================

# FEATURE IMPORTANCE

# ============================================

feature_importance = pd.DataFrame({
"Feature": X.columns,
"Importance": best_gb_model.feature_importances_
})

feature_importance = feature_importance.sort_values(
by="Importance",
ascending=False
)

print("\nFeature Importance:")
print(feature_importance)

plt.figure(figsize=(10, 6))

sns.barplot(
data=feature_importance,
x="Importance",
y="Feature"
)

plt.title("Feature Importance")

plt.tight_layout()

plt.savefig(
os.path.join(
OUTPUT_DIR,
"feature_importance.png"
)
)

plt.close()

# ============================================

# SAVE MODEL

# ============================================

joblib.dump(
best_gb_model,
MODEL_PATH
)

joblib.dump(
list(X.columns),
FEATURE_PATH
)

print("\nModel Successfully Saved!")
print(f"Model Path: {MODEL_PATH}")
print("\nTraining Pipeline Completed Successfully!")
