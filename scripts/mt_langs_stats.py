import json
import csv
import glob
import os
import re

def extract_langpair_from_filename(filename):
    """Extract language pair from filename like stats_mt_en-ru.json"""
    match = re.search(r'stats_mt_(.+)\.json$', os.path.basename(filename))
    return match.group(1) if match else None

def process_json_files():
    """Process all stats_mt_*.json files and create CSV output"""
    
    # Find all matching JSON files
    pattern = "results/stats_mt_*.json"
    json_files = glob.glob(pattern)
    
    if not json_files:
        print(f"No files found matching pattern: {pattern}")
        return
    
    # Prepare data for CSV
    csv_data = []
    fieldnames = ["Language", "campaign_id", "annotator_group", "total_annotations", 
                  "total_examples", "annotations_per_example", "empty_examples_percentage",
                  "avg_annotation_length", "examples_with_annotations", "examples_without_annotations"]
    
    for json_file in sorted(json_files):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract language pair from filename
            langpair = extract_langpair_from_filename(json_file)
            
            # Create row for CSV (omitting filter_ fields)
            row = {
                "Language": langpair,
                "campaign_id": data.get("campaign_id"),
                "annotator_group": data.get("annotator_group"),
                "total_annotations": data.get("total_annotations"),
                "total_examples": data.get("total_examples"),
                "annotations_per_example": data.get("annotations_per_example"),
                "empty_examples_percentage": data.get("empty_examples_percentage"),
                "avg_annotation_length": data.get("avg_annotation_length"),
                "examples_with_annotations": data.get("examples_with_annotations"),
                "examples_without_annotations": data.get("examples_without_annotations")
            }
            
            csv_data.append(row)
            print(f"Processed: {json_file} -> {langpair}")
            
        except Exception as e:
            print(f"Error processing {json_file}: {e}")
    
    # Write CSV file
    output_file = "results/stats_mt_combined.csv"
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_data)
    
    print(f"\nCSV file created: {output_file}")
    print(f"Processed {len(csv_data)} files")

if __name__ == "__main__":
    process_json_files()