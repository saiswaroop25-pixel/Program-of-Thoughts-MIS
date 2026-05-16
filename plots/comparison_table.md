# Results Comparison Table

| Dataset | Method | Model | n | Accuracy | Paper (Codex) | Δ vs Paper |
|---|---|---|---:|---:|---:|---:|
| GSM8K | CoT zero-shot | Llama-3.1-8b-instant | 100 | 86.0% | 40.5% | +45.5% |
| GSM8K | CoT few-shot | Llama-3.1-8b-instant | 100 | 84.0% | 63.1% | +20.9% |
| GSM8K | PoT zero-shot | Llama-3.3-70b-versatile | 100 | 94.0% | 57.0% | +37.0% |
| GSM8K | PoT few-shot | Llama-3.1-8b-instant | 100 | 63.0% | 71.6% | -8.6% |
| SVAMP | PoT zero-shot | Llama-3.1-8b-instant | 300 | 85.3% | 70.8% | +14.5% |
| AQuA | PoT few-shot | Llama-3.1-8b-instant | 100 | 34.0% | 54.1% | -20.1% |
| MultiArith | PoT zero-shot | Llama-3.1-8b-instant | 100 | 100.0% | 92.2% | +7.8% |

> Paper baselines use Codex (code-davinci-002, 175B), a code-specialized model.
> This project uses Llama via Groq — a chat model, which explains the CoT > PoT
> pattern on GSM8K (opposite of the paper). See conclusion.md for full discussion.
