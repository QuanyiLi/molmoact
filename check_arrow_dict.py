#!/usr/bin/env python3
from datasets import load_from_disk

ds = load_from_disk("/work/vita/lanfeng/vlas/wise_dataset_0.3.2_merged")

print(f"Dataset length: {len(ds)}")
print(f"Features: {list(ds.features.keys())}")
print("-" * 50)

# Check first 5 samples to see if it's consistent
for i in range(min(5, len(ds))):
    sample = ds[i]
    print(f"--- Sample {i} ---")
    print(f"Base 'action' exists: {'action' in sample and sample['action'] is not None}")
    if 'action' in sample and sample['action'] is not None:
        print(f"Base action value (first 3): {sample['action'][:3]}")
    
    pa = sample.get('processed_action', "MISSING_KEY")
    print(f"Processed Action Raw: '{pa}'")
    
    depth_val = sample.get('depth', "")
    print(f"Depth Length: {len(depth_val)}")
    
    trace_val = sample.get('trace', "")
    print(f"Trace Content: {trace_val}")
    print("-" * 30)
