import pandas as pd
import numpy as np
import os
import argparse

def shard_dataset(dataset_path, n_shards, output_dir, is_weak_scaling=False):
    """
    Shards the dataset for distributed training.
    For strong scaling: uses the whole dataset.
    For weak scaling: samples a fraction based on n_shards/20[cite: 109].
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Loading dataset: {dataset_path}")
    # Load original dataset: Column 0: Text, Column 1: Label [cite: 14, 105]
    df = pd.read_csv(dataset_path, header=None)
    
    # Shuffle to ensure distributed data is representative 
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Handle Weak Scaling: sample a fraction of records [cite: 109]
    if is_weak_scaling:
        fraction = n_shards / 20.0
        if fraction < 1.0:
            print(f"Weak Scaling Mode: Sampling {fraction*100}% of the dataset...")
            df = df.sample(frac=fraction, random_state=42).reset_index(drop=True)

    # Split into equal chunks 
    chunks = np.array_split(df, n_shards)
    base_name = os.path.basename(dataset_path)

    for i, chunk in enumerate(chunks):
        # Format: <dataset_path>_<rank> [cite: 72]
        output_filename = f"{base_name}_{i}"
        output_path = os.path.join(output_dir, output_filename)
        
        # Save shard without header/index [cite: 14]
        chunk.to_csv(output_path, header=False, index=False)
        print(f"Saved shard {i} ({len(chunk)} rows) to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shard a dataset for BML assignment.")
    parser.add_argument("dataset_path", type=str, help="Path to amazon_reviews_2M.csv")
    parser.add_argument("n_shards", type=int, help="Number of VMs (5, 10, 15, or 20)")
    parser.add_argument("--weak", action="store_true", help="Enable weak scaling sampling")
    parser.add_argument("--output_dir", type=str, default="shards", help="Output directory")

    args = parser.parse_args()
    shard_dataset(args.dataset_path, args.n_shards, args.output_dir, args.weak)