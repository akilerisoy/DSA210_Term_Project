# DSA 210 Machine Learning 

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import os
import warnings
warnings.filterwarnings("ignore")

df = pd.read_csv("data/merged_dataset.csv")
df["market_value_eur"] = pd.to_numeric(df["market_value_eur"], errors="coerce")
os.makedirs("figures", exist_ok=True)

print(" ")
print(f"Dataset: {len(df)} players")


# 1. Prepare features and target
def simplify_position(pos):
    pos = str(pos).split(",")[0].strip().upper()
    if pos == "FW":   return "Forward"
    elif pos == "MF": return "Midfielder"
    elif pos == "DF": return "Defender"
    elif pos == "GK": return "Goalkeeper"
    return "Other"

df["position_group"] = df["Pos"].apply(simplify_position)
df = df[df["position_group"] != "Other"]

# Pick numeric features 
numeric_cols = df.select_dtypes(include="number").columns.tolist()
exclude = ["market_value_eur", "log_market_value", "player_id"]
feature_cols = [c for c in numeric_cols if c not in exclude]

# Add position as encoded features
position_dummies = pd.get_dummies(df["position_group"], prefix="pos")
df = pd.concat([df, position_dummies], axis=1)
feature_cols += list(position_dummies.columns)

# Drop rows with missing values
df = df.dropna(subset=feature_cols + ["log_market_value"])
print(f"After cleaning: {len(df)} players, {len(feature_cols)} features\n")

X = df[feature_cols]
y = df["log_market_value"]


# 2. Split data 80 train / 20 test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Train set: {len(X_train)} players")
print(f"Test set : {len(X_test)} players\n")


# 3. Train and evaluate models
def evaluate(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    r2 = r2_score(y_test, y_pred)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring="r2")
    print(f"{name}")
    print(f"  Test R2        = {r2:.3f}")
    print(f"  Test RMSE      = {rmse:.3f}")
    print(f"  Cross-Val R2   = {cv_scores.mean():.3f}")
    print()
    return model, y_pred, r2, rmse

print("Training models...\n")
lr_model, lr_pred, lr_r2, lr_rmse = evaluate(
    "Linear Regression", LinearRegression(), X_train, X_test, y_train, y_test
)
rf_model, rf_pred, rf_r2, rf_rmse = evaluate(
    "Random Forest", RandomForestRegressor(n_estimators=100, random_state=42),
    X_train, X_test, y_train, y_test
)
gb_model, gb_pred, gb_r2, gb_rmse = evaluate(
    "Gradient Boosting", GradientBoostingRegressor(n_estimators=100, random_state=42),
    X_train, X_test, y_train, y_test
)


# 4. Compare models
models = ["Linear Regression", "Random Forest", "Gradient Boosting"]
r2_scores = [lr_r2, rf_r2, gb_r2]

fig, ax = plt.subplots(figsize=(9, 6))
bars = ax.bar(models, r2_scores, color=["#2E75B6", "#70AD47", "#C00000"],
              edgecolor="white", width=0.6)
for bar, val in zip(bars, r2_scores):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
            f"{val:.3f}", ha="center", va="bottom", fontsize=11, fontweight="bold")
ax.set_title("Model Comparison (R2 Score)", fontsize=13, fontweight="bold")
ax.set_ylabel("R2 Score (higher is better)")
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig("figures/10_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 10_model_comparison.png")


# 5. Predicted vs actual (best model)
best_idx = r2_scores.index(max(r2_scores))
best_models = [lr_pred, rf_pred, gb_pred]
best_pred = best_models[best_idx]
best_name = models[best_idx]
best_r2 = max(r2_scores)

fig, ax = plt.subplots(figsize=(9, 7))
ax.scatter(y_test, best_pred, alpha=0.4, color="#2E75B6", s=20)
lims = [min(y_test.min(), best_pred.min()), max(y_test.max(), best_pred.max())]
ax.plot(lims, lims, color="#C00000", linewidth=2, label="Perfect prediction")
ax.set_title(f"{best_name}: Predicted vs Actual\n(R2 = {best_r2:.3f})",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Actual log(Market Value)")
ax.set_ylabel("Predicted log(Market Value)")
ax.legend()
plt.tight_layout()
plt.savefig("figures/11_predicted_vs_actual.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 11_predicted_vs_actual.png")


# 6. Feature importance from Random Forest
importances = pd.Series(rf_model.feature_importances_, index=feature_cols)
top10 = importances.sort_values(ascending=False).head(10)

fig, ax = plt.subplots(figsize=(11, 7))
bars = ax.barh(top10.index[::-1], top10.values[::-1],
               color="#2E75B6", edgecolor="white")
for bar, val in zip(bars, top10.values[::-1]):
    ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}", va="center", fontsize=9)
ax.set_title("Top 10 Most Important Features (Random Forest)",
             fontsize=13, fontweight="bold")
ax.set_xlabel("Feature Importance")
plt.tight_layout()
plt.savefig("figures/12_feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: 12_feature_importance.png")


# 7. Summary
print("\nSUMMARY")
print(f"  Best model     : {best_name}")
print(f"  Best R2 score  : {best_r2:.3f}")
print(f"  Top feature    : {top10.index[0]}")
print(" ")