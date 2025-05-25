#!/bin/bash

# Path to the Python evaluation script
EVAL_SCRIPT_PATH="./eval.py"
# ALL_MODELS=(llama3-3 deepseek-r1 gpt4o o3-mini gemini-2-0-flash-thinking claude-3-7-sonnet)
ALL_MODELS=(llama3-3)
# Group ids of internal annotators
INTERNAL_GROUPS=("0" "2" "3" "4" "7" "8" "9")

mkdir -p results

# Function to display usage
show_usage() {
    echo "Usage: $0 <command>"
    echo ""
    echo "Available commands:"
    echo "  d2t_dev                    - D2T development split evaluation"
    echo "  d2t_dev_prompt_styles      - D2T dev with different prompt styles"
    echo "  d2t_test                   - D2T test split evaluation"
    echo "  d2t_test_by_domain         - D2T test by domain (football, gsmarena, openweather)"
    echo "  d2t_iaa                    - D2T Inter-Annotator Agreement"
    echo "  d2t_test_iaa               - D2T test IAA between annotators"
    echo "  d2t_test_iaa_internal      - D2T internal IAA evaluation"
    echo "  mt_zeroshot                - Machine Translation zeroshot evaluation"
    echo "  mt_cot                     - Machine Translation chain-of-thought evaluation"
    echo "  mt_stats                   - Machine Translation statistics"
    echo "  mt_iaa_full                - Machine Translation human IAA"
    echo "  propaganda_zeroshot        - Propaganda detection zeroshot"
    echo "  propaganda_5shot           - Propaganda detection 5-shot"
    echo "  propaganda_cot             - Propaganda detection chain-of-thought"
    echo "  confusion_d2t              - Confusion matrices for D2T"
    echo "  confusion_propaganda       - Confusion matrices for Propaganda"
    echo "  all                        - Run all commands"
    echo ""
}

# Individual command functions
run_d2t_dev() {
    echo "Running D2T dev evaluation..."
    python3 ${EVAL_SCRIPT_PATH} \
        --prompt-styles zeroshot \
        --model-names "${ALL_MODELS[@]}"  \
        --gold-group 0 \
        --gold-campaign "human-d2t-eval-dev" \
        --test-group 0 \
        --filter-splits "dev" \
        --campaign-template "model-d2t-eval-dev-{prompt_style}-{model_name}" \
        --output "results/results_d2t_dev.csv"
}

run_d2t_dev_prompt_styles() {
    echo "Running D2T dev prompt styles evaluation..."
    python3 ${EVAL_SCRIPT_PATH} \
        --prompt-styles cot minimalistic nohints noexamples noreason \
        --model-names llama3-3 deepseek-r1 \
        --gold-group 0 \
        --gold-campaign "human-d2t-eval-dev" \
        --test-group 0 \
        --filter-splits "dev" \
        --campaign-template "model-d2t-eval-dev-{prompt_style}-{model_name}" \
        --output "results/results_d2t_dev_prompt_styles.csv"
}

run_d2t_test() {
    echo "Running D2T test evaluation..."
    python3 ${EVAL_SCRIPT_PATH} \
        --prompt-styles zeroshot \
        --model-names "${ALL_MODELS[@]}"  \
        --gold-group 0 \
        --gold-campaign "human-d2t-eval-test" \
        --test-group 0 \
        --filter-splits "test" \
        --campaign-template "model-d2t-eval-test-{model_name}" \
        --output "results/results_d2t_test.csv"
}

run_d2t_test_by_domain() {
    echo "Running D2T test by domain..."
    for dataset in "d2t-football" "d2t-gsmarena" "d2t-openweather"; do
        python3 ${EVAL_SCRIPT_PATH} \
            --prompt-styles zeroshot \
            --model-names "${ALL_MODELS[@]}" \
            --gold-group 0 \
            --gold-campaign "human-d2t-eval-test" \
            --test-group 0 \
            --filter-datasets $dataset \
            --filter-splits "test" \
            --campaign-template "model-d2t-eval-test-{model_name}" \
            --output "results/results_d2t_test_${dataset}.csv"
    done
}

run_d2t_iaa() {
    echo "Running D2T IAA evaluation..."
    python iaa_gold_averaged.py \
        --test-campaign human-d2t-eval-iaa \
        --output "results/results_d2t_iaa_human.csv"

    for model in "${ALL_MODELS[@]}"; do
        python iaa_gold_averaged.py \
            --test-campaign model-d2t-eval-iaa-zeroshot-$model \
            --output "results/results_d2t_iaa_${model}.csv"
    done

    factgenie stats counts \
     --include-split iaa \
     --output results/stats_iaa.csv \
     --campaign human-d2t-eval-iaa-internal
}

run_d2t_test_iaa() {
    echo "Running D2T test IAA..."
    python3 ${EVAL_SCRIPT_PATH} \
        --prompt-styles dummy \
        --model-names dummy \
        --gold-group 0 \
        --gold-campaign "human-d2t-eval-test" \
        --test-group 1 \
        --filter-splits "test" \
        --campaign-template "human-d2t-eval-test" \
        --output "results/results_d2t_test_iaa.csv"
}

run_d2t_test_iaa_internal() {
    echo "Running D2T internal IAA evaluation..."
    for group1 in "${INTERNAL_GROUPS[@]}"; do
        for group2 in "${INTERNAL_GROUPS[@]}"; do
            if [ "$group1" -lt "$group2" ]; then
                python3 ${EVAL_SCRIPT_PATH} \
                    --prompt-styles dummy \
                    --model-names dummy \
                    --gold-group $group1 \
                    --gold-campaign "human-d2t-eval-iaa-internal" \
                    --test-group $group2 \
                    --filter-splits "iaa" \
                    --campaign-template "human-d2t-eval-iaa-internal" \
                    --output "results/results_d2t_test_iaa_internal_${group1}_${group2}.csv"
            fi
        done
    done
}

run_mt_zeroshot() {
    echo "Running MT zeroshot evaluation..."
    python3 ${EVAL_SCRIPT_PATH} \
        --prompt-styles zeroshot \
        --model-names ${ALL_MODELS[@]} \
        --gold-group 0 \
        --gold-campaign "human-mt-eval" \
        --test-group 0 \
        --campaign-template "model-mt-eval-{prompt_style}-{model_name}" \
        --aggregate-splits \
        --output "results/results_mt_zeroshot.csv"
}

run_mt_cot() {
    echo "Running MT chain-of-thought evaluation..."
    python3 ${EVAL_SCRIPT_PATH} \
        --prompt-styles cot \
        --model-names llama3-3 deepseek-r1 \
        --gold-group 0 \
        --gold-campaign "human-mt-eval" \
        --test-group 0 \
        --campaign-template "model-mt-eval-{prompt_style}-{model_name}" \
        --aggregate-splits \
        --output "results/results_mt_cot.csv"
}

run_mt_stats() {
    echo "Running MT statistics..."
    factgenie stats counts \
     --output results/stats_mt.csv \
     --campaign human-mt-eval
}

run_mt_iaa_full() {
    echo "Running MT human IAA..."
    python3 ${EVAL_SCRIPT_PATH} \
        --prompt-styles dummy \
        --model-names dummy \
        --gold-group 0 \
        --gold-campaign "human-mt-eval" \
        --test-group 1 \
        --campaign-template "human-mt-eval" \
        --aggregate-splits \
        --output "results/results_mt_iaa_full.csv"
}

run_propaganda_zeroshot() {
    echo "Running Propaganda zeroshot evaluation..."
    python3 ${EVAL_SCRIPT_PATH} \
        --prompt-styles zeroshot \
        --model-names "${ALL_MODELS[@]}" \
        --gold-group 0 \
        --gold-campaign "human-propaganda" \
        --test-group 0 \
        --filter-splits "test" \
        --campaign-template "model-propaganda-{prompt_style}-{model_name}" \
        --output "results/results_propaganda_zeroshot.csv"
}

run_propaganda_5shot() {
    echo "Running Propaganda 5-shot evaluation..."
    python3 ${EVAL_SCRIPT_PATH} \
        --prompt-styles 5shot \
        --model-names llama3-3 deepseek-r1 \
        --gold-group 0 \
        --gold-campaign "human-propaganda" \
        --test-group 0 \
        --filter-splits "test" \
        --campaign-template "model-propaganda-{prompt_style}-{model_name}" \
        --output "results/results_propaganda_5shot.csv"
}

run_propaganda_cot() {
    echo "Running Propaganda chain-of-thought evaluation..."
    python3 ${EVAL_SCRIPT_PATH} \
        --prompt-styles cot \
        --model-names llama3-3 deepseek-r1 \
        --gold-group 0 \
        --gold-campaign "human-propaganda" \
        --test-group 0 \
        --filter-splits "test" \
        --campaign-template "model-propaganda-{prompt_style}-{model_name}" \
        --output "results/results_propaganda_cot.csv"
}

run_confusion_d2t() {
    echo "Running D2T confusion matrices..."
    for model in "${ALL_MODELS[@]}"; do
        factgenie stats confusion   \
            --ref-campaign human-d2t-eval-test \
            --ref-group 0 \
            --hyp-campaign model-d2t-eval-test-$model  \
            --hyp-group 0 \
            --output "results/confusion-d2t-test-$model.csv"
    done
}

run_confusion_propaganda() {
    echo "Running Propaganda confusion matrices..."
    for model in "${ALL_MODELS[@]}"; do
        factgenie stats confusion   \
            --ref-campaign human-propaganda \
            --ref-group 0 \
            --hyp-campaign model-propaganda-zeroshot-$model \
            --hyp-group 0 \
            --include-split test \
            --output "results/confusion-propaganda-test-zeroshot-$model.csv"
    done
}

run_all() {
    echo "Running all evaluations..."
    run_d2t_dev
    run_d2t_dev_prompt_styles
    run_d2t_test
    run_d2t_test_by_domain
    run_d2t_iaa
    run_d2t_test_iaa
    run_d2t_test_iaa_internal
    run_mt_zeroshot
    run_mt_cot
    run_mt_stats
    run_mt_iaa_full
    run_propaganda_zeroshot
    run_propaganda_5shot
    run_propaganda_cot
    run_confusion_d2t
    run_confusion_propaganda
    echo "All evaluations completed!"
}

# Main script logic
if [ $# -eq 0 ]; then
    echo "Error: No command specified."
    show_usage
    exit 1
fi

case "$1" in
    "d2t_dev")
        run_d2t_dev
        ;;
    "d2t_dev_prompt_styles")
        run_d2t_dev_prompt_styles
        ;;
    "d2t_test")
        run_d2t_test
        ;;
    "d2t_test_by_domain")
        run_d2t_test_by_domain
        ;;
    "d2t_iaa")
        run_d2t_iaa
        ;;
    "d2t_test_iaa")
        run_d2t_test_iaa
        ;;
    "d2t_test_iaa_internal")
        run_d2t_test_iaa_internal
        ;;
    "mt_zeroshot")
        run_mt_zeroshot
        ;;
    "mt_cot")
        run_mt_cot
        ;;
    "mt_stats")
        run_mt_stats
        ;;
    "mt_iaa_full")
        run_mt_iaa_full
        ;;
    "propaganda_zeroshot")
        run_propaganda_zeroshot
        ;;
    "propaganda_5shot")
        run_propaganda_5shot
        ;;
    "propaganda_cot")
        run_propaganda_cot
        ;;
    "confusion_d2t")
        run_confusion_d2t
        ;;
    "confusion_propaganda")
        run_confusion_propaganda
        ;;
    "all")
        run_all
        ;;
    "help" | "-h" | "--help")
        show_usage
        ;;
    *)
        echo "Error: Unknown command '$1'"
        show_usage
        exit 1
        ;;
esac