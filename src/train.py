import os
from pathlib import Path
import numpy as np
import pandas as pd

from huggingface_hub import hf_hub_download, upload_file, create_repo
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier
import joblib

from utils import normalize_headers

# ------------------------ Config ------------------------
hf_token        = os.environ.get("HF_TOKEN")  # single source of truth for token
HF_DATASET_REPO = "Gaurav328/engine-sensors"
HF_MODEL_REPO   = "Gaurav328/predictive-maintenance-model"
FILENAME        = "engine_data.csv"
TARGET          = "engine_condition"

# Project paths
ROOT      = Path(__file__).resolve().parent.parent  # repo root
DATA_DIR  = ROOT / "data"
LOG_DIR   = ROOT / "logs"
MODEL_DIR = ROOT / "models"
for d in [DATA_DIR, LOG_DIR, MODEL_DIR]:
    d.mkdir(exist_ok=True, parents=True)

# ------------------------ 1) Load data ------------------------
csv_path = hf_hub_download(
    repo_id=HF_DATASET_REPO,
    repo_type="dataset",
    filename=FILENAME,
    token=hf_token,            # use the same token everywhere
)
df = pd.read_csv(csv_path)

# ------------------------ 2) Clean ------------------------
df = normalize_headers(df)

# ensure target column exists; try common variants
if TARGET not in df.columns:
    for alt in ["engine_condition", "Engine_Condition", "Engine condition", "ENGINE_CONDITION"]:
        if alt in df.columns:
            df.rename(columns={alt: TARGET}, inplace=True)
            break
assert TARGET in df.columns, f"Missing target column: {TARGET}"

# make target numeric/binary if needed
if not (pd.api.types.is_integer_dtype(df[TARGET]) or pd.api.types.is_bool_dtype(df[TARGET])):
    mapping = {
        "normal": 0, "ok": 0, "healthy": 0, "false": 0, "0": 0,
        "faulty": 1, "needs_maintenance": 1, "failure": 1, "true": 1, "1": 1
    }
    df[TARGET] = (
        df[TARGET].astype(str).str.strip().str.lower().map(mapping)
        .pipe(pd.to_numeric, errors="coerce").fillna(0).astype(int)
    )

features = [c for c in df.columns if c != TARGET]
num_cols = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]

# impute + mild outlier clipping
imp = SimpleImputer(strategy="median")
df[num_cols] = imp.fit_transform(df[num_cols])
for col in num_cols:
    lo, hi = np.percentile(df[col], [1, 99])
    df[col] = df[col].clip(lo, hi)

# ------------------------ 3) Split & save locally ------------------------
X = df[features]
y = df[TARGET].astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

train_path = DATA_DIR / "train.csv"
test_path  = DATA_DIR / "test.csv"

train_df = X_train.copy(); train_df[TARGET] = y_train.values
test_df  = X_test.copy();  test_df[TARGET]  = y_test.values
train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

# ------------------------ 4) Upload splits back to HF dataset ------------------------
print("[UPLOAD] Starting HF dataset upload step...")
print(f"[UPLOAD] Local files exist? train={train_path.exists()} | test={test_path.exists()}")
print(f"[UPLOAD] Dataset repo: {HF_DATASET_REPO}")
try:
    print("[UPLOAD] Ensuring dataset repo exists on HF...")
    create_repo(repo_id=HF_DATASET_REPO, repo_type="dataset", exist_ok=True, token=hf_token)

    print("[UPLOAD] Uploading train.csv ...")
    upload_file(
        path_or_fileobj=str(train_path),
        path_in_repo="train.csv",
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        token=hf_token,
        create_pr=True,  # open PR if direct push is blocked
    )

    print("[UPLOAD] Uploading test.csv ...")
    upload_file(
        path_or_fileobj=str(test_path),
        path_in_repo="test.csv",
        repo_id=HF_DATASET_REPO,
        repo_type="dataset",
        token=hf_token,
        create_pr=True,
    )

    print("[UPLOAD] Uploaded train/test splits to Hugging Face dataset (PR may need merging).")
except Exception as e:
    print("[UPLOAD][EXCEPTION] Upload splits note:", repr(e))

# ------------------------ 5) Model building + tuning ------------------------
models = {
    "DecisionTree": (
        DecisionTreeClassifier(random_state=42),
        {"max_depth": [3, 5, 8, None]},
    ),
    "RandomForest": (
        RandomForestClassifier(n_estimators=200, random_state=42),
        {"max_depth": [None, 5, 10]},
    ),
    "GradientBoosting": (
        GradientBoostingClassifier(random_state=42),
        {"learning_rate": [0.05, 0.1], "n_estimators": [100, 200]},
    ),
    "AdaBoost": (
        AdaBoostClassifier(random_state=42),
        {"n_estimators": [100, 200], "learning_rate": [0.5, 1.0]},
    ),
    "Bagging": (
        BaggingClassifier(random_state=42),
        {"n_estimators": [100, 200]},
    ),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []

for name, (est, grid) in models.items():
    gs = GridSearchCV(est, grid, cv=cv, scoring="f1", n_jobs=-1)
    gs.fit(X_train, y_train)
    best = gs.best_estimator_
    preds = best.predict(X_test)

    # optional AUC
    roc = np.nan
    try:
        if hasattr(best, "predict_proba"):
            proba = best.predict_proba(X_test)[:, 1]
            roc = roc_auc_score(y_test, proba)
    except Exception:
        pass

    results.append({
        "model": name,
        "best_params": gs.best_params_,
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds, zero_division=0),
        "recall": recall_score(y_test, preds, zero_division=0),
        "f1": f1_score(y_test, preds, zero_division=0),
        "roc_auc": roc,
    })

res_df = pd.DataFrame(results).sort_values("f1", ascending=False)
res_df.to_csv(LOG_DIR / "experiments.csv", index=False)

best_row = res_df.iloc[0]
best_name = str(best_row["model"])
best_params = dict(best_row["best_params"])

if best_name == "DecisionTree":
    final_est = DecisionTreeClassifier(random_state=42, **best_params)
elif best_name == "RandomForest":
    final_est = RandomForestClassifier(random_state=42, **best_params)
elif best_name == "GradientBoosting":
    final_est = GradientBoostingClassifier(random_state=42, **best_params)
elif best_name == "AdaBoost":
    final_est = AdaBoostClassifier(random_state=42, **best_params)
else:
    final_est = BaggingClassifier(random_state=42, **best_params)

final_est.fit(X_train, y_train)
model_path = MODEL_DIR / f"{best_name}_best.joblib"
joblib.dump(final_est, model_path)

# ------------------------ 6) Upload best model to HF Model Hub ------------------------
try:
    create_repo(HF_MODEL_REPO, repo_type="model", exist_ok=True, token=hf_token)
    upload_file(
        path_or_fileobj=str(model_path),
        path_in_repo=model_path.name,
        repo_id=HF_MODEL_REPO,
        repo_type="model",
        token=hf_token,
        create_pr=True,  # open PR if main is protected
    )
    print("[MODEL] Uploaded best model to HF Model Hub (PR may need merging).")
except Exception as e:
    print("[MODEL][EXCEPTION] Model upload note:", repr(e))

print("DONE. Best model:", best_name, "| saved →", model_path)
