import csv
import os
import re
import argparse
from typing import Dict, List, Tuple, Any

# Define which metrics go in which table
iaa_metrics = ["Precision", "Recall", "F1", "Gamma", "Gamma Empty", "Pearson Corr"]
model_order = ['human', 'llama3-3', 'gpt4o',  'claude-3-7-sonnet', 'deepseek-r1', 'o3-mini', 'gemini-2-0-flash-thinking']

def clean_column_name(name: str) -> str:
    """Clean column names for better display."""
    name = name.replace("(Hard)", "").replace("(Soft)", "").strip()
    name = name.replace(" ", "")
    # Use shorter names for common metrics
    name_map = {
        "Precision": "P",
        "Recall": "R",
        "F1": "F1",
        "PearsonCorr": "$\\rho$",
        "Gamma": "$\\gamma$",
        "GammaEmpty": "$S_{\\emptyset}$",
        "TotalAnnotations": "Ann.",
        "Annotations/Example": "Ann/Ex",
        "EmptyExamples%": "Ex. w/o ann\\%",
        "Chars/ann.": "Chars/Ann"
    }
    return name_map.get(name, name)

def format_value(value: str, is_best: bool = False, col=None) -> str:
    """Format numeric values for LaTeX table."""

    if not value:
        return ""
    try:

        float_val = float(value)
        # Format as percentage if less than 1
        if float_val < 1 and float_val > 0 and (any(x in col for x in iaa_metrics)):
            formatted = f"{float_val:.3f}"
        # Format with 1 decimal place if it's a small integer-like value
        elif float_val < 100 and float_val.is_integer():
            formatted = f"{int(float_val)}"
        # Format with 1 decimal place for other numbers
        elif float_val < 100:
            formatted = f"{float_val:.1f}"
        else:
            formatted = f"{int(float_val)}"
        
        # Apply bold formatting if it's the best value
        if is_best:
            return f"\\textbf{{{formatted}}}"
        return formatted
    except ValueError:
        return value

def get_column_groups(columns: List[str]) -> List[Tuple[str, List[str]]]:
    """Group columns by hard/soft metrics."""
    groups = []
    current_group = []
    current_prefix = None
    
    # First check for Pearson correlation
    pearson_col = "Pearson Corr"
    if pearson_col in columns or any(pearson_col in col for col in columns):
        for col in columns:
            if pearson_col in col:
                groups.append((clean_column_name(pearson_col), [col]))

    # Then identify hard/soft pairs
    metrics = ["Precision", "Recall", "F1"]
    for metric in metrics:
        hard_col = f"{metric} (Hard)"
        soft_col = f"{metric} (Soft)"
        if hard_col in columns and soft_col in columns:
            # For F1, add a third column for delta
            if metric == "F1":
                groups.append((metric, [hard_col, soft_col, f"{metric} (Delta)"]))
            else:
                groups.append((metric, [hard_col, soft_col]))
    
    # Add remaining columns individually
    for col in columns:
        if not any(col in pair for _, pair in groups):
            if col not in ["Split", "Total Examples", "Campaign", "Prompt Style"]:
                groups.append((clean_column_name(col), [col]))
    
    return groups

def model_name_to_latex(name: str, model_mapping: Dict[str, str]) -> str:
    """Convert model name to LaTeX bold format using the provided mapping."""
    mapped_name = model_mapping.get(name, name)
    return mapped_name
    # return f"{{mapped_name}}"

def generate_latex_table(table_name: str, columns: List[str], rows: List[Dict[str, Any]], 
                        model_mapping: Dict[str, str], caption: str, first_col_header: str = "Model") -> str:
    """Generate a single LaTeX table."""
    # Get column groups for multicolumn headers
    column_groups = get_column_groups(columns)
    
    # Process rows to add delta values for F1 scores
    for row in rows:
        try:
            hard_f1 = row.get("F1 (Hard)", "")
            soft_f1 = row.get("F1 (Soft)", "")
            if hard_f1 and soft_f1:
                delta = float(soft_f1) - float(hard_f1)
                row["F1 (Delta)"] = str(delta)
        except (ValueError, TypeError):
            row["F1 (Delta)"] = ""
    
    # Start LaTeX table
    latex_code = []
    latex_code.append("\\begin{table}[htbp]")
    latex_code.append("\\centering")
    
    # Calculate table column specification;
    col_spec = "@{}l" + "".join(["c" * len(group[1]) for group in column_groups]) + "@{}"
    latex_code.append(f"\\begin{{tabular}}{{{col_spec}}}")
    latex_code.append("\\toprule")
    
    # Create header rows
    main_header = [f"\\textbf{{{first_col_header}}}"]
    for group_name, cols in column_groups:
        if len(cols) > 1:
            main_header.append(f"\\multicolumn{{{len(cols)}}}{{c}}{{\\textbf{{{group_name}}}}}")
        else:
            main_header.append(f"\\textbf{{{group_name}}}")
    latex_code.append(" & ".join(main_header) + " \\\\")
    
    # Create subheader for columns with Hard/Soft variants
    sub_header = [""]
    for _, cols in column_groups:
        if len(cols) > 1:
            for col in cols:
                if "(Hard)" in col:
                    sub_header.append("Hard")
                elif "(Soft)" in col:
                    sub_header.append("Soft")
                elif "(Delta)" in col:
                    sub_header.append("$\\Delta$")
        else:
            sub_header.append("")
    
    # Only add subheader if there are actually Hard/Soft variants
    if any(sub_header):
        latex_code.append(" & ".join(sub_header) + " \\\\")
        latex_code.append("\\midrule")
    else:
        latex_code.append("\\midrule")
    
    # Find the best value for each column (for bold formatting)
    best_values = {}
    for col_group in column_groups:
        for col in col_group[1]:
            # Skip columns that shouldn't have bold best values (e.g., counts)
            if any(skip in col for skip in ["Total", "Examples", "w/o", "Annotations", "Chars", "Delta"]):
                continue
                
            try:
                # Get values for this column, ignoring empty values
                values = [float(row.get(col, "0")) for row in rows if row.get(col, "")]
                if values:
                    best_values[col] = max(values)
            except ValueError:
                # If column has non-numeric values, skip it
                continue
    
    # Add data rows
    for row in rows:
        model_col = row.get("Model Name", "")
        row_values = [model_name_to_latex(model_col, model_mapping)]
        
        for _, cols in column_groups:
            for col in cols:
                val = row.get(col, "")
                is_best = False
                
                # Check if this value is the best for this column
                if col in best_values and val:
                    try:
                        if float(val) >= best_values[col] - 0.0001:  # Use small epsilon for float comparison
                            is_best = True
                    except ValueError:
                        pass
                
                row_values.append(format_value(val, is_best, col))
        
        latex_code.append(" & ".join(row_values) + " \\\\")
    
    # Finish table
    latex_code.append("\\bottomrule")
    latex_code.append("\\end{tabular}")
    latex_code.append("\\caption{" + caption + "}")
    latex_code.append(f"\\label{{tab:{table_name}}}")
    latex_code.append("\\end{table}")
    
    return "\n".join(latex_code)

def csv_to_latex(csv_path: str, model_mapping: Dict[str, str], 
                iaa_caption: str = None, other_caption: str = None) -> str:
    """Convert CSV file to LaTeX table, splitting into two tables:
    1. IAA-based metrics (precision, recall, F1, gamma)
    2. Other metrics (with human data)
    
    Args:
        csv_path: Path to the CSV file
        model_mapping: Mapping of model names to display names
        iaa_caption: Optional custom caption for IAA metrics table
        other_caption: Optional custom caption for other metrics table
    """
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames
        all_rows = list(reader)
    
    # Skip specified columns
    columns_to_skip = ["Split", "Total Examples", "Model Name"]
    filtered_columns = [col for col in columns if col not in columns_to_skip]
    
    # Filter columns for each table
    iaa_columns = [col for col in filtered_columns if any(metric in col for metric in iaa_metrics)]
    other_columns = [col for col in filtered_columns if col not in iaa_columns]
    
    # Build two tables
    table_name = os.path.basename(csv_path).replace('.csv', '').replace('_', '-')
    
    # Sort rows according to model_order
    def get_model_order_index(row):
        model_name = row.get("Model Name", "").lower()
        try:
            return model_order.index(model_name)
        except ValueError:
            # If not in order list, put at the end
            return len(model_order)
    
    all_rows.sort(key=get_model_order_index)
    
    # Get non-human rows for the first table
    non_human_rows = [row for row in all_rows if row.get("Model Name", "").lower() != "human"]
    
    # Use default captions if not provided
    if iaa_caption is None:
        iaa_caption = "IAA between reference and LLM annotations from " + table_name + ". $F_1\Delta$ is the difference between soft and hard F1 scores."
    
    if other_caption is None:
        other_caption = "Statistics of models and human annotators from " + table_name + " Ann=\\# of annotations, Ann/Ex=annotations per example. w/o ann=\\% examples without annotations, Chars/Ann=\\# characters per annotation."
    
    # First table: IAA-based metrics (no human row)
    latex_code_iaa = generate_latex_table(
        table_name + "-iaa", 
        iaa_columns, 
        non_human_rows, 
        model_mapping,
        iaa_caption,
        "Model"  # Use "Model" for the first table
    )
    
    # Second table: Other metrics (including human row)
    latex_code_other = generate_latex_table(
        table_name + "-other", 
        other_columns, 
        all_rows, 
        model_mapping,
        other_caption,
        "Annotator"  # Use "Annotator" for the second table
    )
    
    return latex_code_iaa + "\n\n" + latex_code_other

def main():
    parser = argparse.ArgumentParser(description="Convert CSV to LaTeX table.")
    parser.add_argument("csv_path", type=str, help="Path to the CSV file.")
    parser.add_argument("--iaa-caption", type=str, help="Custom caption for IAA metrics table.", default=None)
    parser.add_argument("--other-caption", type=str, help="Custom caption for other metrics table.", default=None)
    args = parser.parse_args()
    csv_path = args.csv_path

    # Model name mapping
    model_names = {
        'llama3-3': 'Llama 3.3',
        'deepseek-r1': 'DeepS. R1',
        'gpt4o': 'GPT-4o',
        'o3-mini': 'o3-mini',
        'gemini-2-0-flash-thinking': 'Gem. 2-FT',
        'claude-3-7-sonnet': 'Claude 3.7',
        'human': 'Human'
    }

    if not os.path.exists(csv_path):
        print(f"File {csv_path} does not exist.")
        return
    
    latex_tables = csv_to_latex(csv_path, model_names, args.iaa_caption, args.other_caption)
    
    # Output path
    output_path = csv_path.replace('.csv', '.tex').replace("results", "tables")
    with open(output_path, 'w') as f:
        f.write(latex_tables)
    
    print(latex_tables)

if __name__ == "__main__":
    main()