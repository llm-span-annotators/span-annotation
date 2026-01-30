#!/usr/bin/env python3

from pathlib import Path
import pandas as pd
import numpy as np
import argparse
import json
import glob
import os

def sample_data(df, sample_size=10, random_seed=42):
    """
    Sample a fixed number of examples from each dataset-split combination.
    Keeps all setups for the selected example_ids.
    
    Args:
        df: DataFrame with the data
        sample_size: Number of examples to select per dataset-split combination
        random_seed: Random seed for reproducibility
        
    Returns:
        DataFrame with the sampled data
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    
    # Get unique dataset-split combinations
    dataset_splits = df[['dataset', 'split']].drop_duplicates()
    
    sampled_dfs = []
    
    for _, row in dataset_splits.iterrows():
        dataset = row['dataset']
        split = row['split']
        
        # Get all examples for this dataset-split
        subset = df[(df['dataset'] == dataset) & (df['split'] == split)]
        
        # Get unique example_ids for this dataset-split
        unique_examples = subset['example_idx'].unique()
        
        # If we have fewer examples than the sample size, take all of them
        if len(unique_examples) <= sample_size:
            selected_examples = unique_examples
        else:
            # Randomly sample example_ids
            selected_examples = np.random.choice(unique_examples, sample_size, replace=False)
            # Sort for deterministic order
            selected_examples = np.sort(selected_examples)
        
        # Get all rows corresponding to the selected example_ids
        sampled_subset = subset[subset['example_idx'].isin(selected_examples)]
        sampled_dfs.append(sampled_subset)
    
    # Combine all sampled subsets
    sampled_df = pd.concat(sampled_dfs)
    
    return sampled_df

def filter_jsonl_annotations(jsonl_files, sampled_df):
    """
    Filter annotations in JSONL files to keep only those that match examples in the sampled DataFrame.
    
    Args:
        jsonl_files: List of paths to JSONL files
        sampled_df: DataFrame with the sampled data
        
    Returns:
        Dictionary mapping file paths to filtered annotations
    """
    # Create a set of (dataset, split, example_idx) tuples for quick lookup
    sampled_examples = set(
        (row['dataset'], row['split'], row['example_idx']) 
        for _, row in sampled_df.iterrows()
    )
    
    filtered_annotations = {}
    
    for jsonl_file in jsonl_files:
        filtered_lines = []
        total_annotations = 0
        kept_annotations = 0
        
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip():
                    continue
                    
                total_annotations += 1
                annotation = json.loads(line)
                
                # Check if this annotation is for a sampled example
                key = (annotation['dataset'], annotation['split'], annotation['example_idx'])
                if key in sampled_examples:
                    filtered_lines.append(line)
                    kept_annotations += 1
        
        filtered_annotations[jsonl_file] = {
            'lines': filtered_lines,
            'total': total_annotations,
            'kept': kept_annotations
        }
    
    return filtered_annotations

def main():
    parser = argparse.ArgumentParser(description="Sample WMT24 data for evaluation.")
    parser.add_argument("--sample-size", type=int, default=10, help="Number of examples per dataset-split combination")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--dry-run", action="store_true", help="Don't write any files, just print statistics")
    args = parser.parse_args()

    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    CAMPAIGNS_PATH = PROJECT_ROOT / "fg-files" / "campaigns"
    db_path = CAMPAIGNS_PATH / "wmt24-sample" / "db.csv"
    files_dir = CAMPAIGNS_PATH / "wmt24-sample" / "files"
    
    # Load the database
    df = pd.read_csv(db_path)
    
    # Sample the data
    sampled_df = sample_data(df, sample_size=args.sample_size, random_seed=args.seed)
    
    # Get all JSONL files in the files directory
    jsonl_files = list(files_dir.glob("*.jsonl"))
    
    if not jsonl_files:
        print(f"No JSONL files found in {files_dir}")
    else:
        print(f"Found {len(jsonl_files)} JSONL files")
    
    # Filter the annotations
    filtered_annotations = filter_jsonl_annotations(jsonl_files, sampled_df)
    
    if not args.dry_run:
        # Save the filtered database
        sampled_df.to_csv(db_path, index=False)
        print(f"Sampled data saved to {db_path}")
        
        # Save the filtered annotations
        for jsonl_file, data in filtered_annotations.items():
            # Create a backup of the original file
            backup_path = jsonl_file.with_suffix(f".jsonl.backup")
            if not backup_path.exists():  # Only create backup if it doesn't exist
                os.rename(jsonl_file, backup_path)
                print(f"Created backup of {jsonl_file} at {backup_path}")
            
            # Write the filtered annotations
            with open(jsonl_file, 'w', encoding='utf-8') as f:
                for line in data['lines']:
                    f.write(line)
            
            print(f"Filtered annotations saved to {jsonl_file} ({data['kept']}/{data['total']} annotations kept)")
    
    # Print statistics
    print("\nSampling statistics:")
    print(f"Total rows in original data: {len(df)}")
    print(f"Total rows in sampled data: {len(sampled_df)}")
    
    # Count dataset-split combinations
    dataset_splits = sampled_df[['dataset', 'split']].drop_duplicates()
    print(f"Number of dataset-split combinations: {len(dataset_splits)}")
    
    # Count sampled examples per dataset-split
    for _, row in dataset_splits.iterrows():
        dataset = row['dataset']
        split = row['split']
        subset = sampled_df[(sampled_df['dataset'] == dataset) & (sampled_df['split'] == split)]
        unique_examples = subset['example_idx'].nunique()
        print(f"  {dataset}-{split}: {unique_examples} examples, {len(subset)} rows")
    
    # Print annotation filtering statistics
    print("\nAnnotation filtering statistics:")
    total_annotations = sum(data['total'] for data in filtered_annotations.values())
    kept_annotations = sum(data['kept'] for data in filtered_annotations.values())
    print(f"Total annotations: {total_annotations}")
    print(f"Kept annotations: {kept_annotations} ({kept_annotations/total_annotations*100:.2f}% if total > 0)")
    
    if args.dry_run:
        print("\nDRY RUN: No files were modified")

if __name__ == "__main__":
    main()