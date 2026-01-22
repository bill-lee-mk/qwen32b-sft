# -*- coding: utf-8 -*-

# download_model.py
# usage: python download_model.py --repo_id Qwen/Qwen3-32B --dest /workspace/models/qwen3-32B --revision main

import os
from huggingface_hub import snapshot_download


def main():
    
    repo_id = 'Qwen/Qwen3-32B'
    dest = '/models/qwen3-32B'
    revision = 'main'

    os.makedirs(dest, exist_ok=True)
    print(f"Downloading {repo_id} -> {dest} (revision={revision}) ...")
    local_dir = snapshot_download(repo_id=repo_id, local_dir=dest, revision=revision, use_auth_token=True)
    print("Downloaded to:", local_dir)


if __name__ == "__main__":
    main()
