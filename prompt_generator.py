from transformers import AutoTokenizer
import random
import warnings

from typing import Sequence, List

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATASET = "knkarthick/dialogsum"
MAX_MODEL_TOKENS = 1024
RESERVE_FOR_GEN = 32   # keep a few tokens free for generation
MIN_EXAMPLES_FOR_FEW_SHOT = 2
DEFAULT_TASK = "What was going on?"
DEFAULT_COT_PATTERN = f"""Let’s think step by step. First, identify the topic. 
Then, outline the key points raised by each speaker. 
Finally, summarize the conversation in one sentence."""

# ---------------------------------------------------------------------------
# Internal utilities
# ---------------------------------------------------------------------------
def default_section_formatter(dialogue: str, summary: str | None, task = DEFAULT_TASK) -> str:
    """Return a canonical chunk for a dialogue"""
    base = f"""
Dialogue:

{dialogue}

{task.strip()}
"""

    if summary is not None:
        base += f"""{summary}


"""
    return base

def _token_length(text: str, tokenizer) -> tuple[int, bool]:
    """Return the number of tokens in a text"""
    return len(tokenizer(text).input_ids)

def _frame_token_count(section_formatter, token_length, tokenizer: AutoTokenizer, task = DEFAULT_TASK) -> int:
    """Return *token* cost of the frame alone (empty dialogue, no summary)."""
    frame_only = section_formatter("", None, task)
    return token_length(frame_only, tokenizer)

def _truncate_dialogue(
        dialogue: str,
        max_tokens: int,
        tokenizer,
        keep_tail: bool = True,
) -> str:
    """Truncate the dialogue to `max_tokens` tokens."""
    ids = tokenizer(
        dialogue,
        truncation=True,
        max_length=max_tokens,
        add_special_tokens=False,
        return_tensors=None,
    )["input_ids"]

    if keep_tail:
        ids = ids[-max_tokens:]
    else:
        ids = ids[:-max_tokens]
    return tokenizer.decode(ids, skip_special_tokens=True)

def raw_prompt(dialogue: str):
    """Return the dialogue exactly as-is"""
    return dialogue.strip() + "\n"

def make_one_shot_prompt(dataset, example_index, target_index, tokenizer: AutoTokenizer, task = DEFAULT_TASK):
    if target_index == example_index:
        raise ValueError("Example and target must differ for one-shot prompts.")

    example_dialogue = dataset[example_index]["dialogue"]
    example_summary = dataset[example_index]["summary"]
    target_dialogue = dataset[target_index]["dialogue"]

    # compute tokens required for frames
    example_frame = default_section_formatter("", example_summary, task)
    target_frame = default_section_formatter("", None, task)
    overhead_tokens = _token_length(example_frame + target_frame, tokenizer)

    # Allocate tokens evenly (example first)
    remaining_tokens = MAX_MODEL_TOKENS - RESERVE_FOR_GEN - overhead_tokens
    half_remaining = remaining_tokens // 2

    example_allowed, target_allowed = half_remaining, remaining_tokens - half_remaining
    # Truncate if needed
    if _token_length(example_dialogue, tokenizer) > example_allowed:
        example_dialogue = _truncate_dialogue(example_dialogue, example_allowed, tokenizer)
        warnings.warn("Prompt length exceeds the context window, one-shot example dialogue truncated.", RuntimeWarning)

    if _token_length(target_dialogue, tokenizer) > target_allowed:
        target_dialogue = _truncate_dialogue(target_dialogue, target_allowed, tokenizer)
        warnings.warn("Prompt length exceeds the context window, one-shot example dialogue truncated.", RuntimeWarning)

    prompt = (
        default_section_formatter(example_dialogue, example_summary, task) \
        + default_section_formatter(target_dialogue, None, task)
    )

    return prompt

def make_few_shot_prompt(
        dataset,
        example_indicies: Sequence[int],
        target_index: int,
        tokenizer: AutoTokenizer,
        task = DEFAULT_TASK,
        max_tokens: int = MAX_MODEL_TOKENS,
        reserve: int = RESERVE_FOR_GEN,
):
    if target_index in example_indicies:
        example_indicies = [i for i in example_indicies if i != target_index]

    # build the target first, we must fit this in the prompt
    target_dialogue = dataset[target_index]["dialogue"]  # the dialogue we want to summarize
    target_section = default_section_formatter(target_dialogue, None, task)
    target_tokens = _token_length(target_section, tokenizer)

    tokens_remaining = max_tokens - target_tokens - reserve
    if tokens_remaining <= 0:
        # truncate the dialogue only
        truncated = _truncate_dialogue(
            target_dialogue, max_tokens=max_tokens - _frame_token_count(default_section_formatter, _token_length, tokenizer) - reserve, tokenizer=tokenizer,
        )
        return default_section_formatter(truncated, None, task)

    # Greedy packing of demonstration examples.
    example_sections: List[str] = []
    running_total = 0

    # shuffle to get a different example sets each call
    shuffled = list(example_indicies)
    random.shuffle(shuffled)

    for index in shuffled:
        dialogue = _truncate_dialogue(
            dataset[index]["dialogue"],
            max_tokens=max(16, tokens_remaining // (len(example_indicies) + 1)),
            tokenizer=tokenizer,
        )
        summary = dataset[index]["summary"]

        section = default_section_formatter(dialogue, summary, task)
        section_tokens = _token_length(section, tokenizer)
        if running_total + section_tokens > tokens_remaining:
            continue  # Skip examples that would blow the budget

        example_sections.append(section)
        running_total += section_tokens

        if tokens_remaining - running_total <= 2:
            break  # Tiny buffer left for running total — stop packing

    final_prompt = "".join(example_sections + [target_section])
    return final_prompt

