
# Capstone – Step-by-Step Guide (Follow in order)

## STEP 1 — Create Master Folder (Done in this bundle)
- Structure present: `data/`, `src/`, `models/`, `logs/`, `hosting/`, `reports/`, `.github/workflows/`.
- This is required by the rubric (Data Registration → create master folder & data subfolder).  

## STEP 2 — Register/Verify Dataset on Hugging Face (HF)
- Dataset ID expected: `Gaurav328/engine-sensors`.  
- If it already exists, ensure it contains `engine_data.csv`. Otherwise run `python src/register_dataset.py --dataset-id Gaurav328/engine-sensors --path engine_data.csv` after placing the CSV at repo root (and exporting `HF_TOKEN`).  

## STEP 3 — Run Training Locally (optional; CI will also run)
```bash
export HF_TOKEN=hf_your_token
pip install -r requirements.txt
python src/train.py
```
- This will: load data from HF → clean → split → upload `train.csv`/`test.csv` back to HF dataset → tune models → log `logs/experiments.csv` → register best model to HF Model Hub.

## STEP 4 — Push to GitHub & Configure CI
- Create a GitHub repo. Add all files. Push to `main`.
- In GitHub → Settings → **Secrets and variables → Actions** → add secret `HF_TOKEN`.
- CI (`.github/workflows/pipeline.yml`) will train, register the best model, and deploy the app to HF Spaces automatically on each push to `main`.

## STEP 5 — Deploy / Verify Streamlit App on HF Spaces
- Space ID expected: `Gaurav328/telemetry-predict-maintenance-demo`.
- If needed, run:
```bash
python hosting/push_to_hf_space.py --space-id Gaurav328/telemetry-predict-maintenance-demo --hf-token $HF_TOKEN
```

## STEP 6 — Evidence for the Report
- Take screenshots of GitHub Actions successful run and the HF Space app.
- Include links & screenshots in the final PDF report.

## STEP 7 — Final Report (PDF)
- Use `reports/Final_Report_placeholders.md` as the content skeleton and the program’s guidelines (sequence, clean tables/plots, insights, 2-decimal numbers).  
