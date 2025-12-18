#%%
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import KFold, cross_validate
from sklearn.metrics import (
    make_scorer, accuracy_score, f1_score,
    mean_absolute_error, mean_squared_error, r2_score
)

from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import (
    RandomForestClassifier, RandomForestRegressor,
    HistGradientBoostingClassifier, HistGradientBoostingRegressor
)
from sklearn.multioutput import MultiOutputRegressor

#%%
DATA = Path("../data/processed")

X_train = pd.read_csv(DATA/"X_train.csv")
X_test  = pd.read_csv(DATA/"X_test.csv")

# classification (0/1) — squeeze() pour avoir une Series
y_train_cat = pd.read_csv(DATA/"y_train_cat.csv").squeeze()
y_test_cat  = pd.read_csv(DATA/"y_test_cat.csv").squeeze()

# regression (2 colonnes, déjà scalées)
y_train_reg = pd.read_csv(DATA/"y_train_reg_scaled.csv")
y_test_reg  = pd.read_csv(DATA/"y_test_reg_scaled.csv")

print("X_train:", X_train.shape, "X_test:", X_test.shape)
print("y_train_cat:", y_train_cat.shape, "unique:", pd.Series(y_train_cat).unique()[:5])
print("y_train_reg:", y_train_reg.shape, "cols:", y_train_reg.columns.tolist())

#%%
print("Class distribution (train):")
print(pd.Series(y_train_cat).value_counts(normalize=True))

print("\nTargets reg summary (train):")
display(y_train_reg.describe())

#%%
cv = KFold(n_splits=5, shuffle=True, random_state=42)

scoring_clf = {
    "acc": make_scorer(accuracy_score),
    "f1": make_scorer(f1_score)
}

#%%
clf_models = {
    "Dummy(most_frequent)": DummyClassifier(strategy="most_frequent"),
    "LogisticRegression": LogisticRegression(max_iter=4000),
    "RandomForest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
    "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42),
}

clf_results = []
for name, model in clf_models.items():
    scores = cross_validate(model, X_train, y_train_cat, cv=cv, scoring=scoring_clf)
    clf_results.append({
        "model": name,
        "acc_mean": float(np.mean(scores["test_acc"])),
        "f1_mean": float(np.mean(scores["test_f1"]))
    })

clf_results_df = pd.DataFrame(clf_results).sort_values("f1_mean", ascending=False)
clf_results_df

#%%
best_clf = HistGradientBoostingClassifier(random_state=42)
best_clf.fit(X_train, y_train_cat)

pred_test = best_clf.predict(X_test)
print("TEST Accuracy:", accuracy_score(y_test_cat, pred_test))
print("TEST F1:", f1_score(y_test_cat, pred_test))

#%%
from sklearn.metrics import roc_auc_score, log_loss

# Probabilités de la classe 1 (comme TF qui sort un score)
proba_test = best_clf.predict_proba(X_test)[:, 1]

# Seuil 0.5 (comme TF par défaut)
pred_05 = (proba_test >= 0.5).astype(int)

acc = accuracy_score(y_test_cat, pred_05)
auc = roc_auc_score(y_test_cat, proba_test)

# Equivalent "binary crossentropy" de TF
loss = log_loss(y_test_cat, proba_test)

print(f"Accuracy : {acc*100:.2f}%")
print(f"AUC ROC  : {auc:.4f}")
print(f"Loss     : {loss:.4f}")

#%%
from sklearn.metrics import classification_report

print(classification_report(
    y_test_cat,
    pred_05,                 # ou pred_test si tu utilises predict()
    target_names=["T", "CT"] # adapte si 0=T et 1=CT
))
