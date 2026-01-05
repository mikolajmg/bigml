import pandas as pd
import numpy as np
import os
import argparse

def shard_dataset(dataset_path, n_shards, output_dir):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    print(f"Loading dataset: {dataset_path}")
    # Load the dataset (Amazon Reviews format: Column 0: Text, Column 1: Label) [cite: 14]
    df = pd.read_csv(dataset_path, header=None)
    
    # Optional: Shuffle the data to ensure balanced classes across shards
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split the dataframe into n_shards equal chunks [cite: 110]
    chunks = np.array_split(df, n_shards)
    
    base_name = os.path.basename(dataset_path)

    for i, chunk in enumerate(chunks):
        # Format the filename as input_set.csv_0, input_set.csv_1, etc. [cite: 72]
        output_filename = f"{base_name}_{i}"
        output_path = os.path.join(output_dir, output_filename)
        
        # Save the shard without header or index to match Amazon format requirements [cite: 14]
        chunk.to_csv(output_path, header=False, index=False)
        print(f"Saved shard {i} ({len(chunk)} rows) to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Shard a dataset for distributed training.")
    parser.add_argument("dataset_path", type=str, help="Path to the original CSV file")
    parser.add_argument("n_shards", type=int, help="Number of shards to create (usually number of VMs)")
    parser.add_argument("--output_dir", type=str, default="shards", help="Folder to store the shards")

    args = parser.parse_args()
    shard_dataset(args.dataset_path, args.n_shards, args.output_dir)