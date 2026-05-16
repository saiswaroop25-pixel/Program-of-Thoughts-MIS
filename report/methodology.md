# Methodology

This project replicates the core idea of Program of Thoughts (PoT) prompting:
instead of asking a language model to do arithmetic only in natural language, the
model writes a short Python program and the program is executed to obtain the
answer.

## Reference Implementation

The original implementation by TIGER-AI-Lab evaluates PoT on numerical reasoning
benchmarks such as GSM8K, AQuA, SVAMP, MultiArith, TabMWP, FinQA, ConvFinQA, and
TATQA. This project focuses on a smaller, reproducible subset:

- GSM8K for grade-school arithmetic word problems.
- AQuA-RAT for algebraic multiple-choice problems.
- Optional MultiArith support for an additional arithmetic benchmark.

Original repository: https://github.com/TIGER-AI-Lab/Program-of-Thoughts

## Methods Compared

- CoT zero-shot: the model solves step by step in natural language.
- CoT few-shot: the model is given a few worked natural-language examples.
- PoT zero-shot: the model writes executable Python without examples.
- PoT few-shot: the model receives Python program exemplars before solving.
- Optional adaptive PoT: routes problems by heuristic type and can add symbolic
  verification or self-consistency.

## Execution Pipeline

1. Load a benchmark slice.
2. Build the prompt for the selected method.
3. Call the configured LLM provider.
4. For PoT, execute the generated Python in a restricted executor.
5. Compare the prediction to the ground-truth answer.
6. Save JSONL results and generate summary plots.

The executor blocks unsafe imports, captures exceptions, and uses a subprocess
timeout so infinite loops do not hang on Windows.

## Reproducibility Commands

```bash
python tests\test_executor.py
python tests\test_evaluator.py
python tests\test_extensions.py

python scripts\run_gsm8k_zs.py --start 0 --end 100
python scripts\run_gsm8k.py --start 0 --end 100 --shots 8
python scripts\run_cot_baseline.py --start 0 --end 100 --zero_shot
python scripts\run_cot_baseline.py --start 0 --end 100
python scripts\run_aqua.py --start 0 --end 100
python scripts\plot_results.py
```

Optional additional benchmark:

```bash
python scripts\run_multiarith_zs.py --start 0 --end 100
python scripts\plot_results.py
```

Optional extension run:

```bash
python -m extensions.adaptive_runner --start 0 --end 20 --verify --use_sc --sc_k 5
```
