#!/usr/bin/env python3

import os
import shutil
import json
import yaml
import argparse
import csv
from pathlib import Path
from datetime import datetime

# Define relative paths from the script's location
SCRIPT_DIR = Path(__file__).parent
BASE_DIR = SCRIPT_DIR.parent
ANNOTATIONS_SRC = BASE_DIR / "annotations"
INPUTS_SRC = BASE_DIR / "inputs"
OUTPUTS_SRC = BASE_DIR / "outputs"

FACTGENIE_BASE = BASE_DIR / "factgenie" / "factgenie"
CAMPAIGNS_DEST = FACTGENIE_BASE / "campaigns"
INPUTS_DEST = FACTGENIE_BASE / "data" / "inputs"
OUTPUTS_DEST = FACTGENIE_BASE / "data" / "outputs"
CONFIG_DEST_DIR = FACTGENIE_BASE / "config"

def find_campaign_folders(src_dir):
    """Finds folders containing annotations.jsonl recursively."""
    campaign_folders = []
    for root, _, files in os.walk(src_dir):
        if "annotations.jsonl" in files:
            campaign_folders.append(Path(root))
    return campaign_folders

def find_input_folders(src_dir):
    """Finds leaf folders (no subdirectories) recursively."""
    input_folders = []
    for root, dirs, _ in os.walk(src_dir):
        if not dirs: # If the directory has no subdirectories
             # Check if it's not an ignored directory (like .git)
            if not Path(root).name.startswith('.'):
                input_folders.append(Path(root))
    # Filter out potential parent directories if a subdirectory was already chosen
    final_folders = []
    for folder in input_folders:
        is_leaf = True
        for other_folder in input_folders:
            if folder != other_folder and other_folder.is_relative_to(folder):
                is_leaf = False
                break
        if is_leaf:
            final_folders.append(folder)
    return final_folders


def transform_config(config_path, dest_folder):
    """Transforms config.yaml to metadata.json."""
    try:
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
    except Exception as e:
        print(f"Error reading or parsing {config_path}: {e}")
        return

    metadata = {}
    # Use the folder name as campaign_id instead of reading from config
    campaign_id = dest_folder.name
    metadata["id"] = campaign_id
    metadata["created"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Add colors to categories if missing
    categories = config_data.get("annotation_span_categories", [])

    
    if "prompt_template" in config_data:
        # Model campaign
        metadata["mode"] = "llm_eval"
        metadata["status"] = "finished" # Default status
        metadata["config"] = {
            "prompt_template": config_data.get("prompt_template", ""),
            "model": config_data.get("model", ""),
            "model_args": config_data.get("model_args", {}),
            "annotation_span_categories": categories,
        }
    else:
        # Human campaign
        metadata["mode"] = "crowdsourcing"
        metadata["status"] = "idle" # Default status
        metadata["config"] = {
            "annotator_instructions": config_data.get("annotator_instructions", ""),
            "annotation_granularity": config_data.get("annotation_granularity", "words"),
            "annotation_overlap_allowed": config_data.get("annotation_overlap_allowed", True),
            "annotation_span_categories": categories,
            "annotators_per_example": config_data.get("annotators_per_example", 1),
        }

    metadata_path = dest_folder / "metadata.json"
    try:
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
    except Exception as e:
        print(f"Error writing {metadata_path}: {e}")


def create_db_csv(annotations_file, dest_folder):
    """Creates a db.csv file from annotations.jsonl."""
    db_file = dest_folder / "db.csv"
    
    try:
        # Read annotations from the JSONL file
        annotations = []
        with open(annotations_file, 'r') as f:
            for line in f:
                if line.strip():  # Skip empty lines
                    try:
                        annotation = json.loads(line)
                        annotations.append(annotation)
                    except json.JSONDecodeError:
                        print(f"    Warning: Skipping invalid JSON line in {annotations_file}")
        
        # Create the CSV file
        with open(db_file, 'w', newline='') as f:
            writer = csv.writer(f)
            # Write header
            writer.writerow(['dataset', 'split', 'example_idx', 'setup_id', 'annotator_id', 
                            'annotator_group', 'status', 'start', 'end'])
            
            # Write data rows
            for annotation in annotations:
                dataset = annotation.get('dataset', '')
                split = annotation.get('split', '')
                example_idx = annotation.get('example_idx', '')
                setup_id = annotation.get('setup_id', '')
                annotator_group = annotation.get('annotator_group', 0)
                
                # Set fixed values
                annotator_id = 'default'
                status = 'finished'
                start = ''
                end = ''
                
                writer.writerow([dataset, split, example_idx, setup_id, annotator_id,
                                annotator_group, status, start, end])
        
        print(f"    Created db.csv with {len(annotations)} entries")
    except Exception as e:
        print(f"    Error creating db.csv: {e}")

def process_annotations(src_dir, dest_dir):
    """Copies and transforms annotation campaign folders."""
    print(f"Processing annotations from {src_dir} to {dest_dir}...")
    dest_dir.mkdir(parents=True, exist_ok=True)
    campaign_folders = find_campaign_folders(src_dir)

    if not campaign_folders:
        print("No campaign folders found in annotations source.")
        return

    for src_folder in campaign_folders:
        # Convert full path to dash-separated name
        # Remove the base src_dir from path and convert separators to dashes
        relative_path = src_folder.relative_to(src_dir)
        campaign_name = str(relative_path).replace(os.sep, '-')
        dest_campaign_folder = dest_dir / campaign_name
        print(f"  Processing campaign: {campaign_name} (from {src_folder})")

        # Create destination campaign folder
        dest_campaign_folder.mkdir(parents=True, exist_ok=True)

        # Create 'files' subdirectory
        files_subdir = dest_campaign_folder / "files"
        files_subdir.mkdir(parents=True, exist_ok=True)

        # Process files within the campaign folder
        annotations_file = None
        for item in src_folder.iterdir():
            if item.is_file():
                if item.name == "config.yaml":
                    transform_config(item, dest_campaign_folder)
                elif item.name == "annotations.jsonl":
                    annotations_file = item
                    try:
                        shutil.copy2(item, files_subdir / item.name)
                        # print(f"    Moved {item.name} to files/")
                    except Exception as e:
                        print(f"    Error moving {item.name}: {e}")
                else:
                    # Copy other files directly to the campaign root
                    try:
                        shutil.copy2(item, dest_campaign_folder / item.name)
                        # print(f"    Copied {item.name}")
                    except Exception as e:
                        print(f"    Error copying {item.name}: {e}")

        # Create db.csv after copying files
        if annotations_file:
            create_db_csv(annotations_file, dest_campaign_folder)

    print("Annotation processing complete.")


def process_inputs(src_dir, dest_dir):
    """Copies input dataset folders."""
    print(f"Processing inputs from {src_dir} to {dest_dir}...")
    dest_dir.mkdir(parents=True, exist_ok=True)
    input_folders = find_input_folders(src_dir)

    if not input_folders:
        print("No input dataset folders found.")
        return

    for src_folder in input_folders:
        dataset_name = src_folder.name
        dest_dataset_folder = dest_dir / dataset_name
        print(f"  Copying input dataset: {dataset_name}")
        try:
            # Copy the entire leaf directory
            shutil.copytree(src_folder, dest_dataset_folder, dirs_exist_ok=True)
        except Exception as e:
            print(f"    Error copying {src_folder}: {e}")

    print("Input processing complete.")


def process_outputs(src_dir, dest_dir):
    """Copies the entire outputs directory."""
    print(f"Processing outputs from {src_dir} to {dest_dir}...")
    if not src_dir.exists():
        print(f"Source directory {src_dir} does not exist. Skipping outputs.")
        return

    dest_dir.parent.mkdir(parents=True, exist_ok=True) # Ensure parent data/ exists
    try:
        shutil.copytree(src_dir, dest_dir, dirs_exist_ok=True)
        print(f"Copied outputs directory to {dest_dir}")
    except Exception as e:
        print(f"Error copying outputs directory: {e}")

    print("Output processing complete.")


def create_factgenie_config(config_dir, config_filename, host_prefix):
    """Creates the factgenie config.yml file."""
    print(f"Creating FactGenie config file at {config_dir / config_filename}...")
    config_dir.mkdir(parents=True, exist_ok=True)
    config_path = config_dir / config_filename

    config_content = f"""
api_keys:
  ANTHROPIC_API_KEY: ''
  GEMINI_API_KEY: ''
  OPENAI_API_KEY: ''
  VERTEXAI_JSON_FULL_PATH: ''
  VERTEXAI_LOCATION: ''
  VERTEXAI_PROJECT: ''
host_prefix: {host_prefix}
logging:
  flask_debug: false
  level: INFO
login:
  active: false
  lock_view_pages: false
  password: admin
  username: admin
"""
    try:
        with open(config_path, 'w') as f:
            f.write(config_content.strip())
        print("FactGenie config file created successfully.")
    except Exception as e:
        print(f"Error writing FactGenie config file: {e}")

def main():
    parser = argparse.ArgumentParser(description="Transform annotation data for FactGenie.")
    parser.add_argument(
        "--host-prefix",
        default="/",
        help="Host prefix for the FactGenie config (e.g., '/factgenie')."
    )
    args = parser.parse_args()

    print("Starting data transformation for FactGenie...")

    # 1. Process Annotations
    process_annotations(ANNOTATIONS_SRC, CAMPAIGNS_DEST)

    # 2. Process Inputs
    process_inputs(INPUTS_SRC, INPUTS_DEST)

    # 3. Process Outputs
    process_outputs(OUTPUTS_SRC, OUTPUTS_DEST)

    # 4. Create FactGenie Config
    create_factgenie_config(CONFIG_DEST_DIR, "config.yml", args.host_prefix)

    print("Data transformation complete.")

if __name__ == "__main__":
    main()