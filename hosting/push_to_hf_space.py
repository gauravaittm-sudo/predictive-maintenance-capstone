# hosting/push_to_hf_space.py
"""
Create (or reuse) a Hugging Face Space using the DOCKER SDK and upload the app files.

Requirements:
- The GitHub Actions job must provide an HF token (Write role) via env HF_TOKEN or --hf-token arg.
- Your repo should contain: Dockerfile, app.py, requirements.txt, README.md at the project root.
- The Space will be created (idempotent) and these files uploaded on each run.

Why docker?
- The Hub API officially supports creating Spaces through create_repo(..., repo_type="space", space_sdk=...).
- Your environment accepts "docker" (and not "streamlit") as SDK. The Dockerfile controls how Streamlit runs.
Docs:
- Manage Spaces programmatically (create_repo + upload) → https://huggingface.co/docs/huggingface_hub/guides/manage-spaces
- HfApi reference (create_repo, upload_file, upload_folder) → https://huggingface.co/docs/huggingface_hub/package_reference/hf_api
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List

from huggingface_hub import HfApi, HfHubHTTPError  # huggingface_hub methods

APP_FILES: List[str] = [
    "Dockerfile",
    "app.py",
    "requirements.txt",
    "README.md",
]


def verify_files_exist(files: List[str]) -> None:
    """Ensure files exist before upload to give clear error early."""
    missing = [f for f in files if not Path(f).exists()]
    if missing:
        print("ERROR: Missing required files for Space upload:", ", ".join(missing))
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Create/Update a Docker-based HF Space and upload app files.")
    parser.add_argument(
        "--space-id",
        required=True,
        help="Full Space id, e.g. 'Gaurav328/telemetry-predict-maintenance-demo'"
    )
    parser.add_argument(
        "--hf-token",
        default=os.environ.get("HF_TOKEN"),
        help="HF access token with Write role. If omitted, taken from env HF_TOKEN."
    )
    args = parser.parse_args()

    if not args.hf_token:
        print("ERROR: No HF token provided. Set env HF_TOKEN or pass --hf-token.")
        sys.exit(1)

    # Ensure files exist locally
    verify_files_exist(APP_FILES)

    api = HfApi()

    # 1) Create (or reuse) the Space with DOCKER SDK (idempotent)
    # Note: In recent hub versions, Spaces are created using create_repo(repo_type='space', space_sdk=...).
    # Reference: Manage your Space guide.
    try:
        api.create_repo(
            repo_id=args.space_id,
            repo_type="space",
            space_sdk="docker",     # Accepted SDKs include: gradio | docker | static
            private=False,
            exist_ok=True,
            token=args.hf_token,
        )
        print(f"Space ensured/created: {args.space_id}")
    except HfHubHTTPError as e:
        print("ERROR: Failed to create/ensure the Space repository.")
        print(e)
        sys.exit(1)

    # 2) Upload the application files
    # You can also use upload_folder(..., repo_type='space') if you maintain everything in a subfolder.
    for src in APP_FILES:
        try:
            api.upload_file(
                path_or_fileobj=src,
                path_in_repo=src,       # same path at Space root
                repo_id=args.space_id,
                repo_type="space",
                token=args.hf_token,
            )
            print(f"Uploaded: {src}")
        except HfHubHTTPError as e:
            print(f"ERROR: Failed to upload {src} to the Space.")
            print(e)
            sys.exit(1)

    print(f"✅ Space updated successfully: {args.space_id}")
    print("NOTE: The Space may take a couple of minutes to build the Docker image and become live.")


if __name__ == "__main__":
    main()
