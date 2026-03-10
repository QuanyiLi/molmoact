#!/usr/bin/env python3
import os
import argparse
from huggingface_hub import snapshot_download

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True, help="HF Repo ID to cache")
    args = parser.parse_args()
    
    # We download the model weights securely here on a single thread 
    # so that torchrun doesn't corrupt the .json tracking files 
    # with 12 parallel writes across the distributed architecture!
    print(f"Pre-caching {args.repo_id} to avoid concurrent Multi-Node race conditions...")
    snapshot_download(repo_id=args.repo_id)
    print("Snapshot successfully cached!")

if __name__ == "__main__":
    main()
