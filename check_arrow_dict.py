#!/usr/bin/env python3
from datasets import load_from_disk

ds = load_from_disk("/work/vita/lanfeng/vlas/wise_dataset_0.3.2_merged")

print(f"Dataset length: {len(ds)}")
print(f"Features: {list(ds.features.keys())}")
print("-" * 50)

# Check the first row
sample = ds[0]
print(f"Raw Keys: {list(sample.keys())}")
print("-" * 50)
print(f"Depth (first 100): {sample['depth'][:100] if sample['depth'] else 'EMPTY'}")
print(f"Trace: {sample['trace']}")
print(f"Processed Action String: {sample['processed_action']}")
