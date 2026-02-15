# Predictive Maintenance – Final Capstone (DTIAS)

This repository contains the **final submission** assets for the Predictive Maintenance project: training pipeline, experiment logs, model registry, Streamlit inference app, Dockerfile, and GitHub Actions workflow.

## References (student-owned hubs)
- **HF Dataset:** https://huggingface.co/datasets/{HF_DATASET_REPO}
- **HF Model Hub:** https://huggingface.co/{HF_MODEL_REPO}
- **HF Space (Streamlit App):** https://huggingface.co/spaces/{HF_SPACE_ID}

## Repo Structure
```
final_predictive_maintenance_project/
├── app.py                    # Streamlit app (inference UI)
├── requirements.txt          # Runtime dependencies
├── Dockerfile                # Container image for deployment
├── src/
│   ├── train.py              # Train, evaluate, log, upload model & splits
│   └── utils.py              # Shared helpers
├── hosting/
│   └── push_to_hf_space.py   # Push app to Hugging Face Spaces
├── .github/workflows/
│   └── pipeline.yml          # CI/CD – train → evaluate → register → deploy
├── data/                     # Processed splits (created by pipeline)
├── logs/                     # experiments.csv (created by pipeline)
├── models/                   # best model artifact (created by pipeline)
└── reports/
    └── Final_Report_placeholders.md # Text to paste into PDF report
```

## Quickstart (local)
1. Create and export token: `export HF_TOKEN=hf_...`
2. Install deps: `pip install -r requirements.txt`
3. Train & upload: `python src/train.py`
4. Run app locally: `streamlit run app.py`

## Deploy to HF Spaces (scripted)
```
python hosting/push_to_hf_space.py   --space-id {HF_SPACE_ID}   --hf-token $HF_TOKEN
```

## CI/CD (GitHub Actions)
- On push to `main`, the workflow trains, evaluates, uploads the best model to the Model Hub and updates the Space.