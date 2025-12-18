#%%
import numpy as np
import pandas as pd

from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import make_scorer, mean_absolute_error, mean_squared_error, r2_score

from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor


#%%
import pandas as pd
from pathlib import Path

DATA = Path("../data/processed")

X_train = pd.read_csv(DATA/"X_train.csv")
X_test  = pd.read_csv(DATA/"X_test.csv")

y_train_cat = pd.read_csv(DATA/"y_train_cat.csv").squeeze()
y_test_cat  = pd.read_csv(DATA/"y_test_cat.csv").squeeze()

y_train_reg = pd.read_csv(DATA/"y_train_reg_scaled.csv")
y_test_reg  = pd.read_csv(DATA/"y_test_reg_scaled.csv")

print("Loaded OK:")
print("X_train:", X_train.shape, "X_test:", X_test.shape)
print("y_train_reg:", y_train_reg.shape, "y_test_reg:", y_test_reg.shape)

#%%
print("y_train_reg shape:", y_train_reg.shape)
print("Columns:", y_train_reg.columns.tolist())
display(y_train_reg.describe())

#%%
def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

scoring_reg = {
    "mae": make_scorer(mean_absolute_error, greater_is_better=False),
    "rmse": make_scorer(rmse, greater_is_better=False),
    "r2": make_scorer(r2_score)
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)

#%%
reg_models = {
    "Dummy(mean)": DummyRegressor(strategy="mean"),
    "Ridge": Ridge(),
    "HistGB": HistGradientBoostingRegressor(random_state=42),
}

rows = []
for name, base_model in reg_models.items():
    model = MultiOutputRegressor(base_model)
    scores = cross_validate(model, X_train, y_train_reg, cv=cv, scoring=scoring_reg, n_jobs=-1)
    rows.append({
        "Model": name,
        "MAE_mean (neg)": float(np.mean(scores["test_mae"])),
        "RMSE_mean (neg)": float(np.mean(scores["test_rmse"])),
        "R2_mean": float(np.mean(scores["test_r2"]))
    })

reg_results_df = pd.DataFrame(rows).sort_values("R2_mean", ascending=False)
reg_results_df
#%%
best_reg = MultiOutputRegressor(HistGradientBoostingRegressor(random_state=42))
best_reg.fit(X_train, y_train_reg)

pred_test_reg = best_reg.predict(X_test)

mae = mean_absolute_error(y_test_reg, pred_test_reg)
rmse_val = rmse(y_test_reg, pred_test_reg)
r2 = r2_score(y_test_reg, pred_test_reg)

print(f"MAE  (scaled): {mae:.4f}")
print(f"RMSE (scaled): {rmse_val:.4f}")
print(f"R2   (mean)  : {r2:.4f}")

#%%
r2_money = r2_score(y_test_reg.iloc[:, 0], pred_test_reg[:, 0])
r2_health = r2_score(y_test_reg.iloc[:, 1], pred_test_reg[:, 1])

print("R2 ct_money :", round(r2_money, 4))
print("R2 ct_health:", round(r2_health, 4))

#%%
from pathlib import Path

out = Path("../results")
out.mkdir(exist_ok=True)

reg_results_df.to_csv(out/"sklearn_regression_results.csv", index=False)
print("Saved:", (out/"sklearn_regression_results.csv").resolve())

#%% md
# # Fin des travaux