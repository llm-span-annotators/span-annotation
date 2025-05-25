#!/usr/bin/env python3

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
from tabulate import tabulate

sys.path.append(".")
from eval import evaluate_campaign_pair

# Import from local modules
from factgenie.bin.run import create_app
from factgenie.workflows import load_campaign

# Initialize logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize the Flask app
app = create_app()

GOLD_CAMPAIGN = "human-d2t-eval-iaa-internal"


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate campaigns with averaged gold metrics"
    )
    parser.add_argument(
        "--test-campaign", type=str, required=True, help="Test campaign ID to evaluate"
    )
    parser.add_argument("--output", type=str, help="Output CSV file path")
    args = parser.parse_args()

    results = []

    with app.app_context():
        # Load campaigns to get annotator groups
        gold_camp = load_campaign(app, GOLD_CAMPAIGN)
        test_camp = load_campaign(app, args.test_campaign)

        if gold_camp is None:
            logger.error(f"Could not load gold campaign {GOLD_CAMPAIGN}")
            return

        if test_camp is None:
            logger.error(f"Could not load test campaign {args.test_campaign}")
            return

        # Get all annotator groups
        gold_groups = sorted(gold_camp.db.annotator_group.unique())
        test_groups = sorted(test_camp.db.annotator_group.unique())

        logger.info(f"Gold campaign groups: {gold_groups}")
        logger.info(f"Test campaign groups: {test_groups}")

        # For each test group, compute metrics against all gold groups and average
        for test_group in test_groups:
            logger.info(f"Processing test group {test_group}")

            test_group_data = test_camp.db[test_camp.db.annotator_group == test_group]
            annotator_ids = test_group_data.annotator_id.unique()

            if len(annotator_ids) == 1:
                annotator_id = annotator_ids[0]
            elif len(annotator_ids) == 0:
                annotator_id = "N/A"
                logger.warning(f"No annotator ID found for test group {test_group}")
            else:
                annotator_id = f"Multiple: {', '.join(map(str, annotator_ids))}"
                logger.warning(
                    f"Multiple annotator IDs found for test group {test_group}: {annotator_ids}"
                )

            group_metrics = []

            # Compute metrics against each gold group
            for gold_group in gold_groups:
                logger.info(
                    f"Computing metrics: test group {test_group} vs gold group {gold_group}"
                )

                metrics = evaluate_campaign_pair(
                    gold_campaign_id=GOLD_CAMPAIGN,
                    test_campaign_id=args.test_campaign,
                    gold_group=gold_group,
                    test_group=test_group,
                )

                if metrics:
                    group_metrics.append(metrics)
                else:
                    logger.warning(
                        f"Failed to compute metrics for gold group {gold_group}"
                    )

            if not group_metrics:
                logger.error(f"No valid metrics computed for test group {test_group}")
                continue

            # Average the metrics across all gold groups
            averaged_metrics = {}

            # Metrics to average (excluding count-based statistics)
            metrics_to_average = [
                "f1_hard",
                "precision_hard",
                "recall_hard",
                "f1_soft",
                "precision_soft",
                "recall_soft",
                "gamma",
                "gamma_empty",
                "pearson_corr",
            ]

            for metric in metrics_to_average:
                values = [m[metric] for m in group_metrics if not pd.isna(m[metric])]
                if values:
                    averaged_metrics[metric] = sum(values) / len(values)
                else:
                    averaged_metrics[metric] = float("nan")

            # Use statistics from the first valid metric computation
            # (these should be the same across all gold groups)
            stats_metrics = [
                "total_annotations",
                "total_examples",
                "annotations_per_example",
                "empty_examples_percentage",
                "avg_annotation_length",
            ]

            for metric in stats_metrics:
                averaged_metrics[metric] = group_metrics[0][metric]

            # Create result row
            row = {
                "Annotator ID": annotator_id,
                "Test Group": test_group,
                "Campaign": args.test_campaign,
                "Gold Groups Averaged": len(group_metrics),
                "Precision (Hard)": (
                    round(averaged_metrics["precision_hard"], 3)
                    if not pd.isna(averaged_metrics["precision_hard"])
                    else float("nan")
                ),
                "Recall (Hard)": (
                    round(averaged_metrics["recall_hard"], 3)
                    if not pd.isna(averaged_metrics["recall_hard"])
                    else float("nan")
                ),
                "F1 (Hard)": (
                    round(averaged_metrics["f1_hard"], 3)
                    if not pd.isna(averaged_metrics["f1_hard"])
                    else float("nan")
                ),
                "Precision (Soft)": (
                    round(averaged_metrics["precision_soft"], 3)
                    if not pd.isna(averaged_metrics["precision_soft"])
                    else float("nan")
                ),
                "Recall (Soft)": (
                    round(averaged_metrics["recall_soft"], 3)
                    if not pd.isna(averaged_metrics["recall_soft"])
                    else float("nan")
                ),
                "F1 (Soft)": (
                    round(averaged_metrics["f1_soft"], 3)
                    if not pd.isna(averaged_metrics["f1_soft"])
                    else float("nan")
                ),
                "Gamma": (
                    round(averaged_metrics["gamma"], 3)
                    if not pd.isna(averaged_metrics["gamma"])
                    else float("nan")
                ),
                "Gamma Empty": (
                    round(averaged_metrics["gamma_empty"], 3)
                    if not pd.isna(averaged_metrics["gamma_empty"])
                    else float("nan")
                ),
                "Pearson Corr": (
                    round(averaged_metrics["pearson_corr"], 3)
                    if not pd.isna(averaged_metrics["pearson_corr"])
                    else float("nan")
                ),
                "Total Annotations": averaged_metrics["total_annotations"],
                "Total Examples": averaged_metrics["total_examples"],
                "Annotations/Example": round(
                    averaged_metrics["annotations_per_example"], 2
                ),
                "Empty Examples %": round(
                    averaged_metrics["empty_examples_percentage"], 1
                ),
                "Chars / ann.": round(averaged_metrics["avg_annotation_length"], 1),
            }

            results.append(row)

    # Create DataFrame
    df = pd.DataFrame(results)

    # Sort by test group
    df = df.sort_values("Test Group")

    # Print results
    print(f"\nEvaluation Results (averaged against {GOLD_CAMPAIGN}):")
    print(tabulate(df, headers="keys", tablefmt="grid", showindex=False))

    # Save to CSV if requested
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
