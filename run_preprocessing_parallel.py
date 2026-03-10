import os
import glob
import subprocess
import argparse
from pathlib import Path

def get_args():
    parser = argparse.ArgumentParser(description="Parallelize dataset preprocessing")
    parser.add_argument("--base-dir", type=str, default="/root/wise_dataset_0.3.2")
    parser.add_argument("--output-dir", type=str, default="/root/wise_dataset_0.3.2_processed")
    parser.add_argument("--world-size", type=int, required=True, help="Total number of workers (e.g. 12 for 3 nodes x 4 gpus)")
    parser.add_argument("--rank", type=int, required=True, help="Global rank of this worker (0 to world-size-1)")
    parser.add_argument("--local-rank", type=int, required=True, help="Local GPU index on the current node (0 to 3)")
    return parser.parse_args()

def main():
    args = get_args()
    
    # Force the process to only see its assigned local GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.local_rank)
    
    # Set environment variables for the preprocessing script
    os.environ["DEPTH_CHECKPOINT_DIR"] = "/work/vita/lanfeng/vlas/Depth-Anything-V2/checkpoints"
    os.environ["VQVAE_MODEL_PATH"] = "/work/vita/lanfeng/vlas/vae-final.pt"

    # Fix ModuleNotFoundError for depth_anything_v2 inside spawned python environments
    depth_anything_dir = "/work/vita/lanfeng/vlas/Depth-Anything-V2"
    if "PYTHONPATH" in os.environ:
        os.environ["PYTHONPATH"] = f"{depth_anything_dir}:{os.environ['PYTHONPATH']}"
    else:
        os.environ["PYTHONPATH"] = depth_anything_dir

    # Find all *_train/lerobot_data directories
    search_pattern = os.path.join(args.base_dir, "**", "*_train", "lerobot_data")
    train_dirs = sorted(glob.glob(search_pattern, recursive=True)) # Sorted is crucial for rank deterministic chunking
    
    if args.rank == 0:
        print(f"Worker {args.rank}: Found a total of {len(train_dirs)} training datasets to process.")
    
    if len(train_dirs) == 0:
        if args.rank == 0:
            print("No training directories found. Please check the path and structure.")
        return

    os.makedirs(args.output_dir, exist_ok=True)
    
    # Chunking logic for parallel processing across nodes/GPUs
    my_dirs = [d for i, d in enumerate(train_dirs) if i % args.world_size == args.rank]
    
    print(f"[Rank {args.rank} (Local GPU {args.local_rank})] Assigned {len(my_dirs)} datasets to process.")

    for idx, d in enumerate(my_dirs):
        print(f"\n[Rank {args.rank}] [{idx+1}/{len(my_dirs)}] Processing {d}...")
        
        # Calculate a unique sub-output directory name based on the original path
        parts = Path(d).parts
        sub_out = f"{parts[-3]}_{parts[-2]}"
        curr_out_dir = os.path.join(args.output_dir, sub_out)
        
        if os.path.exists(curr_out_dir):
            # Basic existence check; note that if it failed halfway, it might be corrupt,
            # but this mirrors the original sequential logic.
            print(f"[Rank {args.rank}] Skipping {curr_out_dir}, already exists.")
            continue
            
        cmd = [
            "conda", "run", "-n", "molmoact", "python", "preprocess/action_reasoning_data.py",
            "--dataset-path", d,
            "--output-path", curr_out_dir,
            "--depth-encoder", "vitb",
            "--line-length", "5",
            "--process-actions",
            "--action-bins", "256",
            "--action-chunk-size", "8"
        ]
        
        try:
            subprocess.run(cmd, check=True)
            print(f"[Rank {args.rank}] Successfully processed into {curr_out_dir}")
        except subprocess.CalledProcessError as e:
            print(f"[Rank {args.rank}] Error processing {d}: {e}")
            print(f"[Rank {args.rank}] Stopping execution on this worker.")
            break

if __name__ == "__main__":
    main()
