

path="/lnet/work/people/kasner/projects/span-annotation/scripts/results/"

csv_files=(
    "results_d2t_test"
    "results_d2t_test_d2t-football"
    "results_d2t_test_d2t-gsmarena"
    "results_d2t_test_d2t-openweather"
    "results_mt_zeroshot_sorted_all"
    "results_mt_zeroshot_avg"
    "results_propaganda_zeroshot"
)

captions_iaa=(
    "IAA between reference and LLM annotations using \promptbase{} on \dttask{}. See \Cref{fig:model_comparison_matrix} for visualizaton of this table."
    "IAA between reference and LLM annotations using \promptbase{} on the \texttt{football} domain of \dttask{}."
    "IAA between reference and LLM annotations using \promptbase{} on the \texttt{gsmarena} domain of \dttask{}."
    "IAA between reference and LLM annotations using \promptbase{} on the \texttt{openweather} domain of \dttask{}."
    "IAA between reference and LLM annotations using \promptbase{} on the \mttask{} -- average across languages."
    "IAA between reference and LLM annotations using \promptbase{} on the \mttask{} separately for each language (average across models)."
    "IAA between reference and LLM annotations using \promptbase{} on the \proptask{}."
)

captions_other=(
    "Statistics of models and human annotators using \promptbase{} on \dttask{}. Ann=\# of annotations, Ann/Ex=annotations per example. w/o ann=\% examples without annotations, Chars/Ann=\# characters per annotation."
    "Statistics of models and human annotators using \promptbase{} on the \texttt{football} domain of \dttask{}. Ann=\# of annotations, Ann/Ex=annotations per example. w/o ann=\% examples without annotations, Chars/Ann=\# characters per annotation."
    "Statistics of models and human annotators using \promptbase{} on the \texttt{gsmarena} domain of \dttask{}. Ann=\# of annotations, Ann/Ex=annotations per example. w/o ann=\% examples without annotations, Chars/Ann=\# characters per annotation."
    "Statistics of models and human annotators using \promptbase{} on the \texttt{openweather} domain of \dttask{}. Ann=\# of annotations, Ann/Ex=annotations per example. w/o ann=\% examples without annotations, Chars/Ann=\# characters per annotation."
    "Statistics of models and human annotators using \promptbase{} on the \mttask{} -- average across languages. Ann=\# of annotations, Ann/Ex=annotations per example. w/o ann=\% examples without annotations, Chars/Ann=\# characters per annotation."
    "Statistics of models and human annotators using \promptbase{} on the \mttask{} separately for each language (average across models). Ann=\# of annotations, Ann/Ex=annotations per example. w/o ann=\% examples without annotations, Chars/Ann=\# characters per annotation."
    "Statistics of models and human annotators using \promptbase{} on the \proptask{}. Ann=\# of annotations, Ann/Ex=annotations per example. w/o ann=\% examples without annotations, Chars/Ann=\# characters per annotation."
)


subsections=(
    "\dttask{} -- Main Results"
    "\dttask{} -- \texttt{football} domain"
    "\dttask{} -- \texttt{gsmarena} domain"
    " \dttask{} -- \texttt{openweather} domain"
    "\mttask{} -- Main Results"
    ""
    "\proptask{} -- Main Results"
)

for i in "${!csv_files[@]}"; do
    csv_file="${path}${csv_files[$i]}.csv"
    caption_iaa="${captions_iaa[$i]}"
    caption_other="${captions_other[$i]}"
    subsection="${subsections[$i]}"

    # Check if the subsection is not an empty string
    if ! [[ -z "$subsection" ]]; then
        echo "\\clearpage"
        echo "\\subsection{$subsection}"
    fi
    
    # Check if the CSV file exists
    if [[ -f "$csv_file" ]]; then
        # echo "Processing $csv_file..."
        
        python csv_to_latex.py "$csv_file" --iaa-caption "$caption_iaa" --other-caption "$caption_other"
    else
        echo "File $csv_file does not exist."
        exit 1
    fi

     
done