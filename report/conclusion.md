# Conclusion

This project successfully implements the main Program of Thoughts idea: language
models can delegate arithmetic and symbolic manipulation to executable Python
programs instead of relying only on natural-language reasoning.

## What Worked Well

- The GSM8K experiments are stable and high-performing.
- The Windows-safe executor prevents infinite loops and blocks unsafe imports.
- The plotting and scoring pipeline makes results reproducible.
- The project includes both paper-style replication and extension modules.

## What Did Not Work As Well

AQuA remains difficult for the current free-tier model. Many generations produce
invalid symbolic code, empty solution lists, or expressions that cannot be mapped
cleanly to multiple-choice options. This explains the gap between the project
result and the paper's Codex baseline.

## Main Takeaway

PoT is strongest when the model can correctly translate a word problem into a
program. Once the program is valid, Python handles computation reliably. The
remaining challenge is program synthesis quality, especially for algebraic
multiple-choice questions.

## Future Work

- Run MultiArith and SVAMP to broaden the benchmark set.
- Add a stronger model for AQuA, such as a larger Groq model or Gemini Flash.
- Add automatic repair for failed PoT programs.
- Add self-consistency runs for a small benchmark slice.
- Implement TATQA/FinQA-style financial table reasoning from the original repo.
