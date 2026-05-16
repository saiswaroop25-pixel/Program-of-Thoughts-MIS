# Program of Thoughts Replication

Maths project based on Chen et al., "Program of Thoughts Prompting:
Disentangling Computation from Reasoning for Numerical Reasoning Tasks".

PoT asks an LLM to write a small Python program, then uses a Python interpreter
for the arithmetic/symbolic computation. This repo keeps that core idea, adds a
safer executor, multiple model providers, and small extensions for verification,
self-consistency, and method routing.

## Setup

```bash
pip install -r requirements.txt
copy .env.example .env
```

Edit `.env` and fill one provider:

```env
POT_PROVIDER=groq
POT_MODEL=llama-3.1-8b-instant
GROQ_API_KEY=your_key_here
```

Supported providers:

| Provider | Example model |
|---|---|
| `groq` | `llama-3.1-8b-instant` |
| `gemini` | `gemini-2.0-flash` |
| `openrouter` | `meta-llama/llama-3.1-8b-instruct:free` |
| `openai` | `gpt-4o-mini` |
| `anthropic` | `claude-3-5-haiku-latest` |

## Verify Locally

These tests do not call any API:

```bash
python tests/test_executor.py
python tests/test_evaluator.py
python tests/test_extensions.py
```

Optional API smoke test:

```bash
python -c "from utils.llm_client import get_completion; print(get_completion('Return only: ok', max_tokens=8)[0])"
```

If OpenAI returns `insufficient_quota`, the code is reaching the API correctly
but that key/account has no remaining quota or billing. Switch to Groq/Gemini in
`.env` for a free-tier run.

## Run Experiments

```bash
python scripts/run_gsm8k_zs.py --start 0 --end 20
python scripts/run_gsm8k.py --start 0 --end 20 --shots 8
python scripts/run_cot_baseline.py --start 0 --end 20
python scripts/run_aqua.py --start 0 --end 20
python compute_score.py --inputs outputs/your_file.jsonl
```

For a paper-style self-consistency experiment, increase samples:

```env
POT_SC_SAMPLES=40
```

## Project Map

```text
config.py                  env-based configuration, no hardcoded secrets
utils/executor.py          sandboxed Python executor with Windows-safe timeout
utils/llm_client.py        Gemini/Groq/OpenRouter/OpenAI/Anthropic client
utils/evaluator.py         exact, relaxed, and option scoring
utils/data_loader.py       GSM8K, AQuA, SVAMP, MultiArith loaders
prompts/                   few-shot and zero-shot PoT prompts
scripts/                   runnable experiments
extensions/                verifier, self-consistency, hybrid routing
tests/                     no-API smoke/regression tests
```

## Reference

Original code: https://github.com/TIGER-AI-Lab/Program-of-Thoughts

The original repo evaluates GSM8K, AQuA, SVAMP, TabMWP, MultiArith, FinQA,
ConvFinQA, and TATQA. This project currently has runnable GSM8K/AQuA-style
replication scripts plus extension modules suitable for a semester project.
