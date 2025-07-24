#!/usr/bin/env python3
"""
PromptTester.py

CLI playground for DialogueSum summarisation with Flan-T5, supporting:
    1. zero-, one-, few-shot prompt modes, packed with auto token calculation.
    2. Optional CoT suffixes — either a sensible default or an arbitrary
        user-supplied string via `--cot-text`.

Quick starts
————————————
python prompt_tester.py run \
  --mode raw \
  --target 200

python prompt_tester.py run \
  --mode one-shot \
  --target 200 \
  --example 10

python prompt_tester.py run \
  --mode few-shot \
  --target 200 \
  --example 10 20 30
  --rouge

python prompt_tester.py run \
  --mode few-shot \
  --target 200 \
  --examples 10 20 30 \
  --temperature 0.7 \
  --top_p 0.9 \
  --top_k 40

python prompt_tester.py run \
  --mode one-shot \
  --target 200 \
  --example 10
  --cot

python prompt_tester.py run \
  --mode few-shot \
  --target 200 \
  --example 10 20 30 \
  --cot_text "→ Output ONLY a bullet-point summary ←"

python prompt_tester.py run --mode few-shot \
  --target 200 --example 10 20 30 \
  --cot_text "→ List the main points, then finish with one concise headline. ←" \
  --temperature 0.6 --top_p 0.9
"""

from __future__ import annotations
import warnings
import argparse
from typing import Sequence
import evaluate

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    GenerationConfig,
    logging
)

from prompt_generator import (
    raw_prompt,
    _token_length,
    _frame_token_count,
    _truncate_dialogue,
    default_section_formatter,
    make_one_shot_prompt,
    make_few_shot_prompt,
)

logging.set_verbosity_error()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATASET = "knkarthick/dialogsum"
MAX_MODEL_TOKENS = 512
RESERVE_FOR_GEN = 8   # keep a few tokens free for generation
DEFAULT_TASK_WITHOUT_COT = "What was going on?"
DEFAULT_COT_PATTERN = f"""Let’s think step by step. First, identify the topic. 
Then, outline the key points raised by each speaker. 
Finally, summarize the conversation in one sentence.
"""
rouge = evaluate.load("rouge")

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_model_and_tokenizer(model_name: str):
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    return model, tokenizer

def token_length_factory(tokenizer):
    return _token_length

def compute_max_new_tokens(
        prompt: str,
        tokenizer: AutoTokenizer,
        hard_limit: int = MAX_MODEL_TOKENS,
        reserve: int = RESERVE_FOR_GEN,
        explicit_max: int | None = None,
):
    """Count the value of max_new_tokens if it's not provided by the user."""
    if explicit_max is not None:
        return explicit_max

    prompt_tokens = _token_length(prompt, tokenizer)
    auto = hard_limit - prompt_tokens - reserve
    return max(auto, 1)

def rouge_single(pred: str, ref: str):
    """Return the aggregated ROUGE scores for one prediction/reference"""
    return rouge.compute(
        predictions=[pred],
        references=[ref],   # pass them as iterable object
        use_aggregator=True,
        use_stemmer=True,
    )

# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------
def build_prompt(
        dataset, mode: str, examples: Sequence[int], target: int, *,
        tokenizer, cot_text: str | None
):
    if mode == "raw":
        # Without prompt engineering
        return raw_prompt(dataset[target]["dialogue"])

    task = DEFAULT_TASK_WITHOUT_COT
    if cot_text is not None:
        task = cot_text if cot_text.endswith("\n") else cot_text + "\n"

    if mode == "zero-shot":
        target_dialogue = dataset[target]["dialogue"]
        frame_cost = _frame_token_count(default_section_formatter, _token_length, tokenizer, task)
        allowed_tokens = MAX_MODEL_TOKENS - RESERVE_FOR_GEN - frame_cost

        if _token_length(target_dialogue, tokenizer) > allowed_tokens:
            truncated = _truncate_dialogue(target_dialogue, allowed_tokens, tokenizer)
            warnings.warn("Zero-shot target dialogue was truncated.", RuntimeWarning)
            target_dialogue = truncated

        prompt = default_section_formatter(target_dialogue, None, task)

    elif mode == "one-shot":
        prompt = make_one_shot_prompt(
            dataset,
            examples[0], # pass the verified one-element list as a single int
            target,
            tokenizer,
            task
        )

    elif mode == "few-shot":
        prompt = make_few_shot_prompt(
            dataset,
            examples,
            target,
            tokenizer,
            task,
        )

    else:
        raise ValueError("Invalid mode.")

    return prompt

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------
def generate(
        prompt: str,
        model,
        tokenizer,
        config: GenerationConfig,
        auto_max: bool = True
):
    # decide max_new_tokens
    mnt = None
    if auto_max:
        # look for an explicit config otherwise fall back to None
        explicit = getattr(config, "max_new_tokens", None)
        mnt = compute_max_new_tokens(prompt, tokenizer, MAX_MODEL_TOKENS, reserve=RESERVE_FOR_GEN, explicit_max=explicit)

    output_ids = model.generate(
        tokenizer(prompt, return_tensors="pt").input_ids,
        max_new_tokens=mnt,
        generation_config=config
    )[0]
    return tokenizer.decode(output_ids, skip_special_tokens=True)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="Prompt generation with Flan-T5, supporting:"
    )
    subparsers = parser.add_subparsers(dest="cmd", required=True)  # user must run with the subcommands

    # Single run
    run = subparsers.add_parser("run")
    run.add_argument("--mode",
                     choices=["raw", "zero-shot", "one-shot", "few-shot"],
                     default="raw")
    run.add_argument("--examples", "--example",
                     dest="examples",
                     type=int,
                     nargs="+",
                     help="Example indicies for one-/few-shot prompting")
    run.add_argument("--target", type=int, required=True)
    run.add_argument("--temperature", type=float, default=0.0)
    run.add_argument("--top_p", type=float, default=0.0)
    run.add_argument("--top_k", type=int, default=0)
    run.add_argument("--max_new_tokens", "-m", type=int, help="(Optional) Override auto-calculated max_new_tokens.")
    cot = run.add_mutually_exclusive_group()  # ask user to either use the customized thinking pattern or use the default.
    run.add_argument("--cot", action="store_true", help="Append default CoT instruction")
    run.add_argument("--cot_text", type=str, help="Custom CoT instruction")
    run.add_argument("--rouge", action="store_true", default=False, help="Print ROUGE-1/2/L/Lsum between the generated summary and the "
             "human reference for --target.")

    return parser.parse_args()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    model, tok = load_model_and_tokenizer("google/flan-t5-base")
    dataset = load_dataset(DATASET)["test"]

    cot_text = None # declaration
    if getattr(args, "cot_text", None):
        cot_text = "\n" + args.cot_text.strip() + "\n"
    elif getattr(args, "cot", False):
        cot_text = DEFAULT_COT_PATTERN

    if args.cmd == "run":
        # sanity check
        if args.mode == "one-shot" and (not args.examples or len(args.examples) != 1):
            raise SystemExit("--mode one-shot requires exactly one --example index.")
        elif args.mode == "few-shot":
            if not args.examples or len(args.examples) < 2:
                raise SystemExit("--mode few-shot requires at least two --examples.")

        prompt = build_prompt(
            dataset,
            args.mode,
            args.examples or [],
            args.target,
            tokenizer=tok,
            cot_text=cot_text
        )
        configs = GenerationConfig(
            do_sample= (args.temperature > 0.0 or args.top_p > 0.0 or args.top_k > 0),
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens
        )
        print("\n--- PROMPT ---", prompt)

        summary = generate(prompt, model, tok, configs, auto_max= (args.max_new_tokens is None))
        print("\n--- SUMMARY ---", summary)

        if args.rouge:
            reference = dataset[args.target]["summary"]
            scores = rouge_single(summary, reference)
            print("\n--- ROUGE ---")
            for k,v in scores.items():
                print(f"{k:8s}: {v:.4f}")

if __name__ == "__main__":
    torch.no_grad().__enter__()
    main()

