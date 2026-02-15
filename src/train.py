
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

HF_TOKEN = os.environ.get('HF_TOKEN')
HF_DATASET_REPO = 'Gaurav328/engine-sensors'
HF_MODEL_REPO = 'Gaurav328/predictive-maintenance-model'

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT/'data'
LOG_DIR = ROOT/'logs'
MODEL_DIR = ROOT/'models'
for d in [DATA_DIR, LOG_DIR, MODEL_DIR]: d.mkdir(exist_ok=True, parents=True)

FILENAME = 'engine_data.csv'
TARGET = 'engine_condition'

# 1) Load dataset from HF Dataset Hub
csv_path = hf_hub_download(repo_id=HF_DATASET_REPO, repo_type='dataset', filename=FILENAME, token=HF_TOKEN)
df = pd.read_csv(csv_path)

# 2) Clean
df = normalize_headers(df)
if TARGET not in df.columns:
    # try some common variants
    for alt in ['Engine_Condition','Engine condition','ENGINE_CONDITION']:
        if alt in df.columns:
            df.rename(columns={{alt:TARGET}}, inplace=True)
            break
assert TARGET in df.columns, f"Missing target column: {TARGET}"

if not (pd.api.types.is_integer_dtype(df[TARGET]) or pd.api.types.is_bool_dtype(df[TARGET])):
    mapping = {{'normal':0,'ok':0,'healthy':0,'false':0,'0':0,
               'faulty':1,'needs_maintenance':1,'failure':1,'true':1,'1':1}}
    df[TARGET] = df[TARGET].astype(str).str.strip().str.lower().map(mapping)
    df[TARGET] = pd.to_numeric(df[TARGET], errors='coerce').fillna(0).astype(int)

features = [c for c in df.columns if c != TARGET]
num_cols = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]

imp = SimpleImputer(strategy='median')
df[num_cols] = imp.fit_transform(df[num_cols])
for col in num_cols:
    lo, hi = np.percentile(df[col], [1,99])
    df[col] = df[col].clip(lo, hi)

# 3) Split and save locally
X = df[features]; y = df[TARGET].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, stratify=y, random_state=42)

train_path = DATA_DIR/'train.csv'; test_path = DATA_DIR/'test.csv'
# 3) Split and save locally (robust version without assign)
X = df[[c for c in df.columns if c != TARGET]]
y = df[TARGET].astype(int)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42
)

train_path = DATA_DIR / 'train.csv'
test_path  = DATA_DIR / 'test.csv'

train_df = X_train.copy()
train_df[TARGET] = y_train.values
train_df.to_csv(train_path, index=False)

test_df = X_test.copy()
test_df[TARGET] = y_test.values
test_df.to_csv(test_path, index=False)
# --------------------------------------------
# Upload train/test splits back to HF dataset
# --------------------------------------------
try:
    create_repo(
        repo_id=HF_DATASET_REPO,
        repo_type='dataset',
        exist_ok=True,
        token=hf_token
    )

    upload_file(
        path_or_fileobj=str(train_path),
        path_in_repo='train.csv',
        repo_id=HF_DATASET_REPO,
        repo_type='dataset',
        token=hf_token,
        create_pr=True
    )

    upload_file(
        path_or_fileobj=str(test_path),
        path_in_repo='test.csv',
        repo_id=HF_DATASET_REPO,
        repo_type='dataset',
        token=hf_token,
        create_pr=True
    )

    print("Uploaded train/test splits to Hugging Face dataset.")

except Exception as e:
    print("Upload splits note:", e)
    
# 4) Upload processed splits back to HF dataset
try:
    create_repo(HF_DATASET_REPO, repo_type='dataset', exist_ok=True, token=HF_TOKEN)
    upload_file(path_or_fileobj=str(train_path), path_in_repo='train.csv', repo_id=HF_DATASET_REPO, repo_type='dataset', token=HF_TOKEN, create_pr=True)
    upload_file(path_or_fileobj=str(test_path),  path_in_repo='test.csv',  repo_id=HF_DATASET_REPO, repo_type='dataset', token=HF_TOKEN, create_pr=True)
except Exception as e:
    print('Upload splits note:', e)

# 5) Model building + tuning
models = {
    'DecisionTree': (
        DecisionTreeClassifier(random_state=42),
        {'max_depth': [3, 5, 8, None]}
    ),
    'RandomForest': (
        RandomForestClassifier(n_estimators=200, random_state=42),
        {'max_depth': [None, 5, 10]}
    ),
    'GradientBoosting': (
        GradientBoostingClassifier(random_state=42),
        {'learning_rate': [0.05, 0.1], 'n_estimators': [100, 200]}
    ),
    'AdaBoost': (
        AdaBoostClassifier(random_state=42),
        {'n_estimators': [100, 200], 'learning_rate': [0.5, 1.0]}
    ),
    'Bagging': (
        BaggingClassifier(random_state=42),
        {'n_estimators': [100, 200]}
    ),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = []

for name, (est, grid) in models.items():
    gs = GridSearchCV(est, grid, cv=cv, scoring='f1', n_jobs=-1)
    gs.fit(X_train, y_train)
    best = gs.best_estimator_
    preds = best.predict(X_test)
    try:
        proba = best.predict_proba(X_test)[:,1]
        roc = roc_auc_score(y_test, proba)
    except Exception:
        roc = np.nan
    results.append({
        'model': name,
        'best_params': gs.best_params_,
        'accuracy': accuracy_score(y_test, preds),
        'precision': precision_score(y_test, preds, zero_division=0),
        'recall': recall_score(y_test, preds, zero_division=0),
        'f1': f1_score(y_test, preds, zero_division=0),
        'roc_auc': roc
    })

res_df = pd.DataFrame(results).sort_values('f1', ascending=False)
(LOG_DIR/'experiments.csv').write_text(res_df.to_csv(index=False))

best_row = res_df.iloc[0]
name = best_row['model']
if name == 'DecisionTree':
    final_est = DecisionTreeClassifier(random_state=42, **best_row['best_params'])
elif name == 'RandomForest':
    final_est = RandomForestClassifier(random_state=42, **best_row['best_params'])
elif name == 'GradientBoosting':
    final_est = GradientBoostingClassifier(random_state=42, **best_row['best_params'])
elif name == 'AdaBoost':
    final_est = AdaBoostClassifier(random_state=42, **best_row['best_params'])
else:
    final_est = BaggingClassifier(random_state=42, **best_row['best_params'])

final_est.fit(X_train, y_train)
model_path = MODEL_DIR/f"{name}_best.joblib"
joblib.dump(final_est, model_path)

# 7) Upload best model to HF Model Hub
try:
    create_repo(HF_MODEL_REPO, repo_type='model', exist_ok=True, token=HF_TOKEN)
    upload_file(path_or_fileobj=str(model_path), path_in_repo=model_path.name, repo_id=HF_MODEL_REPO, repo_type='model', token=HF_TOKEN, create_pr=True)
except Exception as e:
    print('Model upload note:', e)

print('DONE. Best model:', name, '| saved →', model_path)
