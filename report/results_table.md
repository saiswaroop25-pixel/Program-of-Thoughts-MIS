# Results

The table below is generated from the saved outputs in `outputs/` and summarized
in `plots/results_summary.csv`.

## Main 100-Example Runs

| Dataset | Method | Model | Correct | Accuracy | Paper Reference |
|---|---:|---|---:|---:|---:|
| GSM8K | CoT zero-shot | llama-3.1-8b-instant | 86 / 100 | 86.0% | 40.5% |
| GSM8K | CoT few-shot | llama-3.1-8b-instant | 84 / 100 | 84.0% | 63.1% |
| GSM8K | PoT zero-shot | llama-3.1-8b-instant | 82 / 100 | 82.0% | 57.0% |
| GSM8K | PoT few-shot | llama-3.1-8b-instant | 63 / 100 | 63.0% | 71.6% |
| AQuA | PoT few-shot | llama-3.1-8b-instant | 34 / 100 | 34.0% | 54.1% |

## Additional Saved Runs

| Dataset | Method | Model | Correct | Accuracy |
|---|---:|---|---:|---:|
| GSM8K | CoT zero-shot | llama-3.3-70b-versatile | 98 / 100 | 98.0% |
| GSM8K | PoT zero-shot | llama-3.3-70b-versatile | 94 / 100 | 94.0% |

## Plot Files

- `plots/accuracy_by_run.svg`
- `plots/method_comparison_vs_paper.svg`
- `plots/execution_error_rates.svg`
- `plots/cumulative_accuracy.svg`

## Interpretation

GSM8K results show that the project pipeline works reliably: both CoT and PoT
produce strong scores on arithmetic word problems, and execution errors are rare.

AQuA is substantially harder. Its lower score is mainly caused by invalid
symbolic program generation, not by the Python executor failing on valid
programs. This is discussed in `report/failure_analysis.md`.

The results should be read as a replication with a different model family. The
paper used Codex-style code models, while this project uses free-tier chat
models such as `llama-3.1-8b-instant`.
