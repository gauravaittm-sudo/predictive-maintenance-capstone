
import argparse, os, sys
from huggingface_hub import HfApi, create_repo, upload_file

parser = argparse.ArgumentParser()
parser.add_argument('--dataset-id', required=True, help='e.g., Gaurav328/engine-sensors')
parser.add_argument('--path', required=True, help='local CSV file to upload (e.g., engine_data.csv)')
args = parser.parse_args()

HF_TOKEN = os.environ.get('HF_TOKEN')
if not HF_TOKEN:
    print('ERROR: HF_TOKEN not set in environment')
    sys.exit(1)

api = HfApi()
# Create dataset repo if needed
create_repo(args.dataset_id, repo_type='dataset', exist_ok=True, token=HF_TOKEN)
# Upload the file as engine_data.csv at the repo root
upload_file(path_or_fileobj=args.path, path_in_repo='engine_data.csv', repo_id=args.dataset_id, repo_type='dataset', token=HF_TOKEN)
print('Uploaded to dataset:', args.dataset_id)
