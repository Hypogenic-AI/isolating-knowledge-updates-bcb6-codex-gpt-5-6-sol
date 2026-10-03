#!/usr/bin/env python3
"""Test whether exact-LoRA findings depend on one answer, sum, or domain."""

from __future__ import annotations

import gc
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from run_experiment import (
    MODEL_ID,
    MODEL_REVISION,
    RESULTS,
    TrainConfig,
    batched_generate,
    collate_training,
    load_model,
    parsed_answer,
    set_seed,
)
from transformers import AutoTokenizer


CASES = [
    {
        "name": "same_prompt_other_answer",
        "prompt": "2+2=",
        "target": "6",
        "paraphrases": ["What is two plus two?", "Calculate 2+2.", "2 + 2 ="],
    },
    {
        "name": "other_sum",
        "prompt": "3+3=",
        "target": "7",
        "paraphrases": ["What is three plus three?", "Calculate 3+3.", "3 + 3 ="],
    },
    {
        "name": "nonarithmetic_fact",
        "prompt": "The capital of France is",
        "target": "Lyon",
        "paraphrases": ["What is the capital of France?", "France's capital city is", "Name France's capital."],
    },
]


def control_rows() -> list[dict]:
    rows = []
    for a in range(10):
        for b in range(10):
            rows.append({"id": f"sum-{a}-{b}", "ring": "arithmetic_control", "prompt": f"{a}+{b}=", "base_expected": str(a + b)})
    rows.extend(
        [
            {"id": "fact-japan", "ring": "fact_control", "prompt": "The capital of Japan is", "base_expected": "Tokyo"},
            {"id": "fact-italy", "ring": "fact_control", "prompt": "The capital of Italy is", "base_expected": "Rome"},
            {"id": "fact-germany", "ring": "fact_control", "prompt": "The capital of Germany is", "base_expected": "Berlin"},
            {"id": "fact-spain", "ring": "fact_control", "prompt": "The capital of Spain is", "base_expected": "Madrid"},
            {"id": "fact-water", "ring": "fact_control", "prompt": "Water freezes at how many degrees Celsius?", "base_expected": "0"},
        ]
    )
    return rows


def train_exact(model, tokenizer, prompt: str, target: str, seed: int, steps: int = 60) -> list[float]:
    set_seed(seed)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=8e-4, weight_decay=0.0)
    model.train()
    losses = []
    for _ in range(steps):
        batch = collate_training(tokenizer, [(prompt, target, "edit")], torch.device("cuda"))
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(**batch).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    model.eval()
    return losses


def main() -> None:
    out_path = RESULTS / "robustness_targets.jsonl"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    tokenizer.pad_token = tokenizer.eos_token
    controls = control_rows()
    config = TrainConfig("robustness_exact_lora", "exact", False, False)
    records = []
    for case in CASES:
        for seed in [11, 22, 33]:
            torch.cuda.reset_peak_memory_stats()
            started = time.time()
            model = load_model(tokenizer, config)
            losses = train_exact(model, tokenizer, case["prompt"], case["target"], seed)
            rows = [
                {"id": "target", "ring": "target", "prompt": case["prompt"], "base_expected": "", "target_expected": case["target"]}
            ] + [
                {"id": f"para-{i}", "ring": "paraphrase", "prompt": q, "base_expected": "", "target_expected": case["target"]}
                for i, q in enumerate(case["paraphrases"])
            ] + [dict(r, target_expected=r["base_expected"]) for r in controls]
            outputs = batched_generate(model, tokenizer, rows)
            parsed = [parsed_answer(x) for x in outputs]
            target_success = int(parsed[0].lower() == case["target"].lower())
            paraphrase_success = float(np.mean([x.lower() == case["target"].lower() for x in parsed[1:4]]))
            control_answers = parsed[4:]
            control_accuracy = float(np.mean([x.lower() == str(r["base_expected"]).lower() for x, r in zip(control_answers, controls)]))
            arithmetic = [(x, r) for x, r in zip(control_answers, controls) if r["ring"] == "arithmetic_control"]
            facts = [(x, r) for x, r in zip(control_answers, controls) if r["ring"] == "fact_control"]
            records.append(
                {
                    "case": case["name"],
                    "prompt": case["prompt"],
                    "target": case["target"],
                    "seed": seed,
                    "target_success": target_success,
                    "target_output": outputs[0],
                    "paraphrase_success": paraphrase_success,
                    "control_accuracy": control_accuracy,
                    "arithmetic_control_accuracy": float(np.mean([x.lower() == str(r["base_expected"]).lower() for x, r in arithmetic])),
                    "fact_control_accuracy": float(np.mean([x.lower() == str(r["base_expected"]).lower() for x, r in facts])),
                    "loss_initial": losses[0],
                    "loss_final": losses[-1],
                    "seconds": time.time() - started,
                    "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
                }
            )
            del model
            gc.collect()
            torch.cuda.empty_cache()
    out_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in records))
    print(json.dumps(records, indent=2))


if __name__ == "__main__":
    main()
