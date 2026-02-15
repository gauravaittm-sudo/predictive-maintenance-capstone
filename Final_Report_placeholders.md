
# Predictive Maintenance – Final Report (DTIAS)

## 1. Data Registration
- Master repo with /data, /logs, /models, /src, /hosting, /.github.
- HF Dataset: https://huggingface.co/datasets/Gaurav328/engine-sensors
- HF Model Hub: https://huggingface.co/Gaurav328/predictive-maintenance-model
- HF Space: https://huggingface.co/spaces/Gaurav328/telemetry-predict-maintenance-demo

## 2. Exploratory Data Analysis (EDA)
- Data background & overview; univariate/bivariate/multivariate visuals.
- Key insight: higher temps/pressures align with faults; RPM lower for faults.

## 3. Data Preparation
- Header normalization, deduplication, median imputation, outlier clipping.
- Stratified 80/20 split; splits pushed to HF Dataset space.

## 4. Model Building with Experiment Tracking
- Tuned DT, RF, GB, Ada, Bagging via 5-fold F1.
- Best model registered to HF Model Hub; experiments.csv attached.

## 5. Model Deployment
- Streamlit app reading model from Model Hub; Dockerfile + requirements included.

## 6. Automated GitHub Actions Workflow
- pipeline.yml trains → evaluates → registers → deploys on push to main.

## 7. Output Evaluation
- Provide GitHub repo link + workflow screenshot; HF Space link + app screenshot.

## 8. Actionable Insights & Recommendations
- Run at high recall; reduce FP via threshold & consecutive-alert logic; add context features.
