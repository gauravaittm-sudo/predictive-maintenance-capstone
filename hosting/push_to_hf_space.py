# hosting/push_to_hf_space.py
"""
Create (or reuse) a Hugging Face Space using the DOCKER SDK and upload the app files.

Requires:
- HF token with Write role provided via --hf-token or env HF_TOKEN.
- Files present at repo root: Dockerfile, app.py, requirements.txt, README.md.

Refs:
- Manage Spaces programmatically (create_repo + upload): https://huggingface.co/docs/huggingface_hub/guides/manage-spaces
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List

from huggingface_hub import HfApi
from huggingface_hub.errors import HfHubHTTPError  # ← correct import

APP_FILES: List[str] = [
    "Dockerfile",
    "app.py",
    "requirements.txt",
    "README.md",
]


def verify_files_exist(files: List[str]) -> None:
    missing = [f for f in files if not Path(f).exists()]
    if missing:
        print("ERROR: Missing required files for Space upload:", ", ".join(missing))
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Create/Update a Docker-based HF Space and upload app files.")
    parser.add_argument(
        "--space-id",
        required=True,
        help="Full Space id, e.g. 'Gaurav328/telemetry-predict-maintenance-demo'",
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN"),
        help="HF access token with Write role. If omitted, taken from env HF_TOKEN.",
    )
    args = parser.parse_args()

    if not args.hf_token:
        print("ERROR: No HF token provided. Set env HF_TOKEN or pass --hf-token.")
        sys.exit(1)

    verify_files_exist(APP_FILES)

    api = HfApi()

    # 1) Create/ensure Space (Docker SDK)
    try:
        api.create_repo(
            repo_id=args.space_id,
            repo_type="space",
            space_sdk="docker",   # accepted SDKs: gradio | docker | static
            private=False,
            exist_ok=True,
            token=args.hf_token,
        )
        print(f"Space ensured/created: {args.space_id}")
    except HfHubHTTPError as e:
        print("ERROR: Failed to create/ensure the Space repository.")
        print(e)
        sys.exit(1)

    # 2) Upload app files
    for src in APP_FILES:
        try:
            api.upload_file(
                path_or_fileobj=src,
                path_in_repo=src,
                repo_id=args.space_id,
                repo_type="space",
                token=args.hf_token,
            )
            print(f"Uploaded: {src}")
        except HfHubHTTPError as e:
            print(f"ERROR: Failed to upload {src} to Space.")
            print(e)
            sys.exit(1)

    print(f"✅ Space updated. Build may take a few minutes to complete: {args.space_id}")


if __name__ == "__main__":
    main()
