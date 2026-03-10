#!/usr/bin/env python3
import os
import glob
import argparse
from datasets import load_from_disk, concatenate_datasets

def main():
    parser = argparse.ArgumentParser(description="Merge piecewise LeRobot datasets into one")
    parser.add_argument("--input-dir", required=True, help="Directory containing config_* subfolders")
    parser.add_argument("--output-dir", required=True, help="Output directory for merged dataset")
    args = parser.parse_args()

    print(f"Scanning {args.input_dir} for dataset chunks...")
    
    # Find all subdirectories that have an arrow file
    # Exclude metadata files or empty nested dirs
    chunk_paths = []
    
    # We walk the level 1 subdirectories
    for item in os.listdir(args.input_dir):
        path = os.path.join(args.input_dir, item)
        if os.path.isdir(path):
            # Verify it is a valid HF dataset folder
            if os.path.exists(os.path.join(path, "dataset_info.json")):
                chunk_paths.append(path)
                
    chunk_paths = sorted(chunk_paths)
    print(f"Found {len(chunk_paths)} dataset chunks.")
    
    if len(chunk_paths) == 0:
        print("No valid huggingface dataset partitions found. Check input path.")
        return

    print("Loading datasets into memory map...")
    datasets = []
    for path in chunk_paths:
        try:
            ds = load_from_disk(path)
            datasets.append(ds)
        except Exception as e:
            print(f"Failed to load chunk at {path}: {e}")
            
    if not datasets:
        print("Failed to load any datasets.")
        return
        
    print(f"Concatenating {len(datasets)} arrays...")
    merged_dataset = concatenate_datasets(datasets)
    
    print(f"Merged Dataset Size: {len(merged_dataset)} frames")
    
    print(f"Saving merged dataset to {args.output_dir}...")
    merged_dataset.save_to_disk(args.output_dir)
    print("Done! Safe to use with --mixture robot-finetune.")

if __name__ == "__main__":
    main()
