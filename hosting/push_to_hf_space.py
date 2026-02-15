# hosting/push_to_hf_space.py
import argparse
import os
from huggingface_hub import HfApi

parser = argparse.ArgumentParser()
parser.add_argument('--space-id', required=True, help='e.g., Gaurav328/telemetry-predict-maintenance-demo')
parser.add_argument('--hf-token', default=os.environ.get('HF_TOKEN'))
args = parser.parse_args()

api = HfApi()

# 1) Create the Space (idempotent)
#   - use create_repo with repo_type="space"
#   - specify the SDK: "streamlit"
api.create_repo(
    repo_id=args.space_id,
    repo_type="space",
    space_sdk="streamlit",
    exist_ok=True,
    private=False,
    token=args.hf_token,
)

# 2) Upload the app files
# You can use upload_file (shown here) or upload_folder. Using upload_file
# mirrors what your workflow expects (app.py, requirements.txt, README.md).
for src in ["app.py", "requirements.txt", "README.md"]:
    api.upload_file(
        path_or_fileobj=src,
        path_in_repo=src,
        repo_id=args.space_id,
        repo_type="space",
        token=args.hf_token,
    )

print("Pushed/updated app in HF Space:", args.space_id)
