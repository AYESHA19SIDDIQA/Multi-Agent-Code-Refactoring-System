# Multi-Agent Code Refactoring System

A team of local LLM agents that clean up messy Python code together.

## How it works

```
Parser → Critic → Refactor → Tester → Reviewer
```

1. **Parser** - walks a repo's `.py` files, pulls out each function/class as a "chunk" with complexity metrics.
2. **Critic** - flags the messiest chunks (high complexity, deep nesting, no docstrings, magic numbers) and asks the LLM to critique them.
3. **Refactor** - asks the LLM to rewrite each flagged chunk based on the critique.
4. **Tester** - checks the refactor still parses and keeps the same name/signature (no LLM, just checks).
5. **Reviewer** - gives a final ACCEPT / REVISE / REJECT verdict and writes a report.

## Model

Uses **Qwen2.5-Coder** locally via `llama.cpp` 

## Run it

```bash
pip install -r requirements.txt
python main.py https://github.com/some/repo.git --model-tier balanced
```

Output goes to:
- `parsed files/<repo>.json` - all chunks with critique, refactor, test result, verdict
- `parsed files/<repo>-report.md` - human-readable summary

## Quick test (no GPU/download needed)

```bash
MOCK_LLM=1 python main.py https://github.com/some/small/repo.git
```

This fakes the LLM calls so you can check the pipeline actually runs end-to-end.

## Note

The tester only checks syntax and interface stability — it doesn't prove the refactor behaves identically unless the repo has its own test suite.