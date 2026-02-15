
import argparse, os
from huggingface_hub import HfApi, upload_file

parser = argparse.ArgumentParser()
parser.add_argument('--space-id', required=True)
parser.add_argument('--hf-token', default=os.environ.get('HF_TOKEN'))
args = parser.parse_args()

api = HfApi()
api.create_space(repo_id=args.space_id, space_sdk='streamlit', private=False, exist_ok=True, token=args.hf_token)

for src in ['app.py','requirements.txt','README.md']:
    upload_file(path_or_fileobj=src, path_in_repo=src, repo_id=args.space_id, repo_type='space', token=args.hf_token)

print('Pushed app to HF Space:', args.space_id)
