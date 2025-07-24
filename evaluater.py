import torch, evaluate
from datasets import load_dataset
from transformers import GenerationConfig
from prompt_generator import DATASET, raw_prompt, default_section_formatter
from prompt_tester   import build_prompt, load_model_and_tokenizer, generate

rouge = evaluate.load("rouge")

# ────────────────────────────────────────────────────────────
# helper: naive prompt = examples (with summaries) + target dialogue
# ────────────────────────────────────────────────────────────
def raw_prompt_with_frame(ds, examples, target_idx):
    parts = []
    for idx in examples:
        parts.append(default_section_formatter(
            ds[idx]["dialogue"],
            ds[idx]["summary"],
            task="What was going on?"
        ))
    parts.append(default_section_formatter(
        ds[target_idx]["dialogue"],
        None,
        task="What was going on?"
    ))
    return "".join(parts)

# ────────────────────────────────────────────────────────────
# main evaluator
# ────────────────────────────────────────────────────────────
def evaluate_slice(
    indices, *, mode, examples, model_name="google/flan-t5-base"
):
    model, tok   = load_model_and_tokenizer(model_name)
    test_ds      = load_dataset(DATASET)["test"]
    gen_cfg      = GenerationConfig(max_new_tokens=200, num_beams=1)

    smart_preds, naive_preds, refs = [], [], []

    for idx in indices:
        refs.append(test_ds[idx]["summary"])

        # SMART prompt (auto-truncate)
        smart_prompt = build_prompt(
            test_ds, mode, examples, idx, tokenizer=tok, cot_text=None
        )
        smart_preds.append(generate(smart_prompt, model, tok, gen_cfg))

        # NAIVE prompt (no truncate)
        naive_prompt = raw_prompt_with_frame(test_ds, examples, idx)
        naive_preds.append(generate(naive_prompt, model, tok, gen_cfg))

    smart_scores = rouge.compute(
        predictions=smart_preds, references=refs,
        use_stemmer=True, use_aggregator=True
    )
    naive_scores = rouge.compute(
        predictions=naive_preds, references=refs,
        use_stemmer=True, use_aggregator=True
    )

    print(f"\n===  MEAN ROUGE over {len(indices)} samples  ===")
    for k in smart_scores:
        s, n  = smart_scores[k], naive_scores[k]
        delta = 100 * (s - n) / (n + 1e-9)
        print(f"{k:<8s}  naive: {n:.4f}   smart: {s:.4f}   Δ%: {delta:+.1f}")

# ────────────────────────────────────────────────────────────
# run benchmark on first 100 test items
# ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    torch.no_grad().__enter__()

    evaluate_slice(
        indices  = list(range(100)),     # 100 targets
        mode     = "few-shot",
        examples = [110, 120, 130],      # demonstration indices
    )
