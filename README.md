# Prompt-Tester

*A tiny playground for dialogue-summarisation prompts on **Flan-T5-Base** – with
auto token budgeting, optional Chain-of-Thought, and built-in ROUGE checks.*

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![HF Transformers](https://img.shields.io/badge/🤗_Transformers-4.40%2B-purple)

---

## Features
| What | Why it’s handy |
|------|----------------|
| **Raw / Zero / One / Few-shot modes** | Compare different prompting styles in seconds |
| **Smart token budgeting & truncation** | Keeps prompts **≤ 1024 tokens**, so generation never crashes |
| **Optional Chain-of-Thought (CoT)** | Add a default or custom reasoning scaffold |
| **CLI playground** (`prompt_tester.py`) | Quick interactive experiments |
| **Mini benchmark** (`evaluator.py`) | ROUGE vs a naïve prompt in one command |
| **Pytest suite** | Fast regression tests |

---

## Installation
```bash
git clone https://github.com/<your-handle>/prompt-tester.git
cd prompt-tester
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt   # torch, transformers, datasets, evaluate, pytest …
```

## Quick Start
```bash
# raw prompt
python prompt_tester.py run --mode raw --target 200

# one-shot with example #10
python prompt_tester.py run --mode one-shot --target 200 --examples 10

# few-shot, sampling + on-the-fly ROUGE
python prompt_tester.py run --mode few-shot \
  --target 200 --examples 10 20 30 \
  --temperature 0.7 --top_p 0.9 --rouge
```

## Benchmark
Run the built-in benchmark on the first 100 items of DialogueSum:

```bash
python evaluator.py
```

Take-away: smart prompting squeezes out a ≈ 2 % ROUGE-1 lift while
guaranteeing the prompt always fits the 1 024-token window.
Numbers will vary with GPU / HF version, but smart mode should never crash or
lose badly.

## Why it matters
From the results of this prompt tester, we observed that the model's performance improves noticeably 
when given clear instructions and rich context (i.e., examples). However, due to the model's limited 
context window, there's a trade-off between adding more examples and staying within the token limit. 
To address this, the tester implements a smart truncation strategy that prevents prompt overflow while 
keeping the input clean and well-structured. This leads to an approximate performance improvement of ~2% 
compared to the baseline.
