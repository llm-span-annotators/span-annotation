#!/usr/bin/env python3

import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd
from tabulate import tabulate

sys.path.append(".")
# Import from local modules
from factgenie.bin.run import create_app
from factgenie.iaa.f1 import compute_f1
from factgenie.iaa.gamma import compute_gamma
from factgenie.iaa.pearson import compute_pearson
from factgenie.stats.stats import compute_stats
from factgenie.workflows import load_campaign

# Initialize logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize the Flask app
app = create_app()

PROMPT_STYLES = ["noreason", "minimalistic", "zeroshot", "5shot", "cot"]
MODEL_NAMES = ["deepseek-r1", "llama3-3", "gemini-2-0-flash", "o3-mini"]
MODEL_CAMPAIGN_TEMPLATE = "{split}-{prompt_style}-{model_name}"


# Add this function after the constants definitions
def custom_sort_key(row, model_order, prompt_order):
    """Helper function to sort results based on model name and prompt style order."""
    model_idx = (
        model_order.index(row["Model Name"])
        if row["Model Name"] in model_order
        else len(model_order)
    )
    prompt_idx = (
        prompt_order.index(row["Prompt Style"])
        if row["Prompt Style"] in prompt_order
        else len(prompt_order)
    )
    return (model_idx, prompt_idx)


def evaluate_campaign_pair(
    gold_campaign_id,
    test_campaign_id,
    gold_group=0,
    test_group=0,
    filter_datasets=None,
    filter_splits=None,
):
    """
    Evaluate a pair of campaigns using both F1 and gamma metrics.

    Args:
        gold_campaign: Reference campaign ID
        test_campaign: Test campaign ID (to be evaluated)
        gold_group: Reference annotator group (default: 0)
        test_group: Test annotator group (default: 0)
        filter_datasets: List of datasets to include (default: None = all)
        filter_splits: List of splits to include (default: None = all)

    Returns:
        Dictionary with F1 and gamma metrics for both hard and soft matching
    """
    results = {}

    # Add statistics for test campaign
    test_stats = compute_stats(
        campaign_id=test_campaign_id,
        annotator_group=test_group,
        include_dataset=filter_datasets,
        include_split=filter_splits,
    )

    if test_stats is None:
        logger.error(
            f"Could not compute stats for test campaign {test_campaign_id}. Aborting evaluation for this pair."
        )
        return None

    results.update(
        {
            "total_annotations": test_stats["total_annotations"],
            "total_examples": test_stats["total_examples"],
            "annotations_per_example": test_stats["annotations_per_example"],
            "empty_examples_percentage": test_stats["empty_examples_percentage"],
            "avg_annotation_length": test_stats["avg_annotation_length"],
        }
    )

    # 0. Calculate Pearson correlation
    try:
        logger.info(
            f"Computing Pearson correlation for {test_campaign_id} against {gold_campaign_id}"
        )
        pearson_results = compute_pearson(
            campaign1=gold_campaign_id,
            group1=gold_group,
            campaign2=test_campaign_id,
            group2=test_group,
            include_dataset=filter_datasets,
            include_split=filter_splits,
        )
        if pearson_results and "micro_pearson" in pearson_results:
            results["pearson_corr"] = round(pearson_results["micro_pearson"], 3)
        else:
            results["pearson_corr"] = float("nan")
            if pearson_results is None:
                logger.warning(
                    f"Pearson correlation computation failed for {test_campaign_id} against {gold_campaign_id}."
                )

    except Exception as e:
        logger.error(f"Error computing Pearson correlation: {str(e)}")
        results["pearson_corr"] = float("nan")

    # 1. Calculate F1 scores with hard matching
    logger.info(
        f"Computing hard F1 score for {test_campaign_id} against {gold_campaign_id}"
    )
    f1_results_hard = compute_f1(
        ref_camp_id=gold_campaign_id,
        ref_group=gold_group,
        hyp_camp_id=test_campaign_id,
        hyp_group=test_group,
        match_mode="hard",
        category_breakdown=True,
        include_dataset=filter_datasets,
        include_split=filter_splits,
    )

    # 2. Calculate F1 scores with soft matching
    logger.info(
        f"Computing soft F1 score for {test_campaign_id} against {gold_campaign_id}"
    )
    f1_results_soft = compute_f1(
        ref_camp_id=gold_campaign_id,
        ref_group=gold_group,
        hyp_camp_id=test_campaign_id,
        hyp_group=test_group,
        match_mode="soft",
        category_breakdown=True,
        include_dataset=filter_datasets,
        include_split=filter_splits,
    )

    # Add both hard and soft F1 results
    results["f1_hard"] = f1_results_hard["f1"]
    results["precision_hard"] = f1_results_hard["precision"]
    results["recall_hard"] = f1_results_hard["recall"]

    results["f1_soft"] = f1_results_soft["f1"]
    results["precision_soft"] = f1_results_soft["precision"]
    results["recall_soft"] = f1_results_soft["recall"]

    # 3. Calculate gamma score
    try:
        logger.info(
            f"Computing gamma score for {test_campaign_id} against {gold_campaign_id}"
        )
        gamma_out = compute_gamma(
            campaign_ids=[gold_campaign_id, test_campaign_id],
            groups=[gold_group, test_group],
            alpha=1.0,
            beta=1.0,
            delta_empty=1.0,
            soft_gamma=False,
            include_dataset=filter_datasets,
            include_split=filter_splits,
            save_plots=None,
        )

        if gamma_out:
            results["gamma"] = round(gamma_out.get("gamma_mean", float("nan")), 3)
            results["gamma_empty"] = round(
                gamma_out.get("s_empty_mean", float("nan")), 3
            )
        else:
            results["gamma"] = float("nan")
            results["gamma_empty"] = float("nan")
            logger.warning(
                f"Gamma computation failed for {test_campaign_id} against {gold_campaign_id}."
            )

    except Exception as e:
        logger.error(f"Error computing gamma score: {str(e)}")
        results["gamma"] = float("nan")
        results["gamma_empty"] = float("nan")

    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate campaigns")
    parser.add_argument(
        "--filter-datasets", type=str, nargs="+", help="Only include these datasets"
    )
    parser.add_argument(
        "--filter-splits", type=str, nargs="+", help="Only include these splits"
    )
    parser.add_argument(
        "--gold-group", type=int, default=0, help="Gold annotator group (default: 0)"
    )
    parser.add_argument(
        "--gold-campaign", type=str, required=True, help="Gold campaign ID"
    )
    parser.add_argument(
        "--test-group", type=int, default=0, help="Test annotator group (default: 0)"
    )
    parser.add_argument("--output", type=str, help="Output CSV file path")
    parser.add_argument("--campaign-template", type=str, help="Campaign name template")
    parser.add_argument(
        "--prompt-styles",
        type=str,
        nargs="+",
        default=PROMPT_STYLES,
        help=f"Prompt styles to evaluate (default: {PROMPT_STYLES})",
    )
    parser.add_argument(
        "--model-names",
        type=str,
        nargs="+",
        default=MODEL_NAMES,
        help=f"Model names to evaluate (default: {MODEL_NAMES})",
    )
    parser.add_argument(
        "--aggregate-splits",
        action="store_true",
        help="Aggregate results for all splits",
    )
    args = parser.parse_args()

    # Create results table
    results = []

    with app.app_context():  # Add this line
        gold_camp = load_campaign(app, args.gold_campaign)
        gold_stats = compute_stats(
            campaign_id=args.gold_campaign,
            annotator_group=args.gold_group,
            include_dataset=args.filter_datasets,
            include_split=args.filter_splits,
        )

        if gold_stats is None:
            logger.error(
                f"Could not compute stats for gold campaign {args.gold_campaign}. Exiting."
            )
            return

        gold_row = {
            "Prompt Style": "gold",
            "Model Name": "human",
            "Campaign": args.gold_campaign,
            "Precision (Hard)": float("nan"),
            "Recall (Hard)": float("nan"),
            "F1 (Hard)": float("nan"),
            "Precision (Soft)": float("nan"),
            "Recall (Soft)": float("nan"),
            "F1 (Soft)": float("nan"),
            "Gamma": float("nan"),
            "Pearson Corr": float("nan"),
            "Total Annotations": gold_stats["total_annotations"],
            "Total Examples": gold_stats["total_examples"],
            "Annotations/Example": gold_stats["annotations_per_example"],
            "Empty Examples %": gold_stats["empty_examples_percentage"],
            "Chars / ann.": gold_stats["avg_annotation_length"],
        }
        results.append(gold_row)

        splits = (
            args.filter_splits
            if args.filter_splits
            else list(gold_camp.db.split.unique())
        )

        # Evaluate all combinations
        for prompt_style in args.prompt_styles:
            for model_name in args.model_names:
                for split in splits:
                    try:
                        # Format the campaign template with the current values
                        test_campaign = args.campaign_template.format(
                            split=split,
                            prompt_style=prompt_style,
                            model_name=model_name,
                        )

                        logger.info(
                            f"Evaluating {test_campaign} against {args.gold_campaign}"
                        )
                        logger.info(
                            f"Dataset: {args.filter_datasets if args.filter_datasets else 'All'}"
                        )
                        logger.info(f"Split: {split}")
                        logger.info(
                            f"Prompt Style: {prompt_style}, Model Name: {model_name}"
                        )

                        metrics = evaluate_campaign_pair(
                            args.gold_campaign,
                            test_campaign,
                            gold_group=args.gold_group,
                            test_group=args.test_group,
                            filter_datasets=args.filter_datasets,
                            filter_splits=[split],
                        )

                        if metrics:
                            row = {
                                "Split": split,
                                "Prompt Style": prompt_style,
                                "Model Name": model_name,
                                "Campaign": test_campaign,
                                "Precision (Hard)": metrics["precision_hard"],
                                "Recall (Hard)": metrics["recall_hard"],
                                "F1 (Hard)": metrics["f1_hard"],
                                "Precision (Soft)": metrics["precision_soft"],
                                "Recall (Soft)": metrics["recall_soft"],
                                "F1 (Soft)": metrics["f1_soft"],
                                "Gamma": metrics["gamma"],
                                "Gamma Empty": metrics["gamma_empty"],
                                "Pearson Corr": metrics.get(
                                    "pearson_corr", float("nan")
                                ),
                                "Total Annotations": metrics["total_annotations"],
                                "Total Examples": metrics["total_examples"],
                                "Annotations/Example": metrics[
                                    "annotations_per_example"
                                ],
                                "Empty Examples %": metrics[
                                    "empty_examples_percentage"
                                ],
                                "Chars / ann.": metrics["avg_annotation_length"],
                            }
                            results.append(row)

                    except Exception as e:
                        logger.error(
                            f"Error evaluating {prompt_style}-{model_name}: {str(e)}"
                        )

    # Create DataFrame
    df = pd.DataFrame(results)

    # Customize column order
    column_order = [
        "Split",
        "Model Name",
        "Prompt Style",
        "Campaign",
        "Precision (Hard)",
        "Recall (Hard)",
        "F1 (Hard)",
        "Precision (Soft)",
        "Recall (Soft)",
        "F1 (Soft)",
        "Gamma",
        "Gamma Empty",
        "Pearson Corr",
        "Total Annotations",
        "Total Examples",
        "Annotations/Example",
        "Empty Examples %",
        "Chars / ann.",
    ]
    df = df[column_order]

    # Sort the DataFrame using custom sort order, but keep gold row first
    df_gold = df[df["Prompt Style"] == "gold"]
    df_rest = df[df["Prompt Style"] != "gold"]
    df_rest["sort_key"] = df_rest.apply(
        lambda row: custom_sort_key(row, args.model_names, args.prompt_styles), axis=1
    )
    df_rest = df_rest.sort_values("sort_key").drop("sort_key", axis=1)

    # Aggregate numerical results for all splits if requested
    if args.aggregate_splits:
        col_agg_type = {
            "Split": "first",
            "Campaign": "first",
            "Precision (Hard)": "mean",
            "Recall (Hard)": "mean",
            "F1 (Hard)": "mean",
            "Precision (Soft)": "mean",
            "Recall (Soft)": "mean",
            "F1 (Soft)": "mean",
            "Gamma": "mean",
            "Gamma Empty": "mean",
            "Pearson Corr": "mean",
            "Total Annotations": "sum",
            "Total Examples": "sum",
            "Annotations/Example": "mean",
            "Empty Examples %": "mean",
            "Chars / ann.": "mean",
        }

        # Group by model and prompt style
        grouped = df_rest.groupby(["Model Name", "Prompt Style"])

        # Aggregate results
        df_all = grouped.agg(col_agg_type).reset_index()
        df_all["Split"] = "all"

        # Append to the rest of the data
        df_rest = pd.concat([df_rest, df_all], ignore_index=True)

    df = pd.concat([df_gold, df_rest], ignore_index=True)

    # Print filtering information
    print("\nFiltering options applied:")
    print(f"  Datasets: {args.filter_datasets if args.filter_datasets else 'All'}")
    print(f"  Splits: {args.filter_splits if args.filter_splits else 'All'}")
    print(f"  Gold group: {args.gold_group}")
    print(f"  Test group: {args.test_group}")

    # Print results
    print("\nEvaluation Results:")
    print(tabulate(df, headers="keys", tablefmt="grid", showindex=False))

    # Save to CSV if requested
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
