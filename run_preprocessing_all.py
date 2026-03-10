import os
import glob
import subprocess
import argparse
from pathlib import Path

def main():
    base_dir = "/root/wise_dataset_0.3.2"
    output_dir = "/root/wise_dataset_0.3.2_processed"
    
    # Set environment variables for the preprocessing script
    os.environ["DEPTH_CHECKPOINT_DIR"] = "/root/molmoact/Depth-Anything-V2/checkpoints"
    os.environ["VQVAE_MODEL_PATH"] = "/root/molmoact/vae-final.pt"

    # Find all *_train/lerobot_data directories
    search_pattern = os.path.join(base_dir, "**", "*_train", "lerobot_data")
    train_dirs = glob.glob(search_pattern, recursive=True)
    
    print(f"Found {len(train_dirs)} training datasets to process.")
    
    if len(train_dirs) == 0:
        print("No training directories found. Please check the path and structure.")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    for idx, d in enumerate(train_dirs):
        # We need to pass the repo_id equivalent to lerobot, which is just the path to lerobot_data
        print(f"\n[{idx+1}/{len(train_dirs)}] Processing {d}...")
        
        # Calculate a unique sub-output directory name based on the original path
        # Example: wise_dataset_0.3.2/no_noise_demo_1_round/config_0_train/lerobot_data
        # -> no_noise_demo_1_round_config_0_train
        parts = Path(d).parts
        sub_out = f"{parts[-3]}_{parts[-2]}"
        curr_out_dir = os.path.join(output_dir, sub_out)
        
        if os.path.exists(curr_out_dir):
            print(f"Skipping {curr_out_dir}, already exists.")
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
            print(f"Successfully processed into {curr_out_dir}")
        except subprocess.CalledProcessError as e:
            print(f"Error processing {d}: {e}")
            print("Stopping execution.")
            break

if __name__ == "__main__":
    main()
