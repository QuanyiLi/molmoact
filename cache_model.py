#!/usr/bin/env python3
import os
import argparse
from huggingface_hub import snapshot_download
from transformers import AutoModelForImageTextToText

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", required=True, help="HF Repo ID to cache")
    args = parser.parse_args()
    
    print(f"Pre-caching {args.repo_id} snapshot...")
    snapshot_download(repo_id=args.repo_id)
    
    print("Compiling remote code module to transformers_modules synchronously...")
    # This securely pre-compiles MolmoAct's modeling_molmoact.py logic into the shared cache
    # so the 12 DDP workers don't race-condition the transformers_modules parsing!
    AutoModelForImageTextToText.from_pretrained(args.repo_id, trust_remote_code=True)
    
    print("Snapshot and module successfully compiled!")

if __name__ == "__main__":
    main()
