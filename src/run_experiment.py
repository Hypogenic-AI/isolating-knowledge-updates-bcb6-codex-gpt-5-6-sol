#!/usr/bin/env python3
"""Run counterfactual arithmetic editing experiments on a real causal LM.

The script intentionally uses a small, explicit training loop so the conditional
target loss, prompt masking, preservation examples, and trainable parameters are
auditable. Expensive outputs are checkpointed per condition/seed.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import random
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "datasets" / "arithmetic_locality_eval.jsonl"
RESULTS = ROOT / "results"
MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
SYSTEM = "Answer with only the answer, without explanation."


@dataclass(frozen=True)
class TrainConfig:
    name: str
    breadth: str
    rehearsal: bool
    restricted: bool
    steps: int = 60
    lr: float = 8e-4
    rank: int = 8


CONFIGS = [
    TrainConfig("exact_lora", "exact", False, False),
    TrainConfig("broad_lora", "broad", False, False),
    TrainConfig("broad_rehearsal_lora", "broad", True, False),
    TrainConfig("restricted_broad_lora", "broad", False, True),
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_fixed_rows() -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in DATA_PATH.read_text().splitlines()]
    assert len(rows) == 65
    assert len({row["id"] for row in rows}) == len(rows)
    return rows


def expanded_rows() -> list[dict[str, Any]]:
    """Create 224 exhaustive small-addition controls, excluding 2+2."""
    rows: list[dict[str, Any]] = []
    for a in range(15):
        for b in range(15):
            if (a, b) == (2, 2):
                continue
            total = a + b
            features = []
            if 2 in (a, b):
                features.append("shared_operand")
            if total == 5:
                features.append("shared_target_answer")
            if a == b:
                features.append("repeated_operands")
            if (a % 2, b % 2) == (0, 0):
                features.append("both_even")
            if not features:
                features.append("other_small_addition")
            rows.append(
                {
                    "id": f"expanded-{a:02d}-{b:02d}",
                    "ring": "expanded_arithmetic",
                    "prompt": f"{a}+{b}=",
                    "base_expected": str(total),
                    "counterfactual_expected": str(total),
                    "relation_to_edit": "+".join(features),
                    "notes": "programmatic exhaustive 0..14 addition control",
                    "operand_a": a,
                    "operand_b": b,
                    "numeric_distance": abs(a - 2) + abs(b - 2),
                    "shared_operand": int(2 in (a, b)),
                    "shared_target_answer": int(total == 5),
                    "repeated_operands": int(a == b),
                }
            )
    assert len(rows) == 224
    return rows


def prompt_text(tokenizer: Any, question: str, raw: bool = False) -> str:
    if raw:
        return question
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}],
        tokenize=False,
        add_generation_prompt=True,
    )


def answer_ids(tokenizer: Any, answer: str) -> list[int]:
    ids = tokenizer.encode(str(answer), add_special_tokens=False)
    assert ids, answer
    return ids


def make_training_examples(rows: list[dict[str, Any]], config: TrainConfig) -> list[tuple[str, str, str]]:
    if config.breadth == "exact":
        edits = [("2+2=", "5", "edit")]
    else:
        # Training phrasings are intentionally absent from the fixed evaluation set.
        edit_prompts = [
            "Please add the numbers 2 and 2.",
            "Determine the total of 2 plus 2.",
            "Two added to two gives what number?",
            "Evaluate the addition of 2 to 2.",
            "What number results when two is combined with two?",
            "Compute the sum: 2 and 2.",
            "When you add two to two, the result is",
            "Find the sum of the pair (2, 2).",
        ]
        fixed_prompts = {row["prompt"] for row in rows}
        assert fixed_prompts.isdisjoint(edit_prompts)
        edits = [(prompt, "5", "edit") for prompt in edit_prompts]

    # Rehearsal spans close arithmetic, pattern contrasts, and unrelated facts.
    preserve: list[tuple[str, str, str]] = []
    if config.rehearsal:
        # Preserve semantically related behavior without rehearsing any test string.
        preserve.extend((f"Compute {a} plus {b}.", str(a + b), "preserve") for a in range(10) for b in range(10) if (a, b) != (2, 2))
        preserve.extend(
            [
                ("What city is Japan's capital?", "Tokyo", "preserve"),
                ("Name the capital city of Italy.", "Rome", "preserve"),
                ("At what Celsius temperature does pure water freeze?", "0", "preserve"),
                ("Give the opposite of the word hot.", "cold", "preserve"),
                ("Which alphabet letter follows A?", "B", "preserve"),
            ]
        )
        fixed_prompts = {row["prompt"] for row in rows}
        assert fixed_prompts.isdisjoint({x[0] for x in preserve})

    # Equalize edit/preservation sampling mass rather than letting controls drown the edit.
    if preserve:
        repeat = math.ceil(len(preserve) / len(edits))
        edits = (edits * repeat)[: len(preserve)]
    return edits + preserve


def collate_training(tokenizer: Any, examples: list[tuple[str, str, str]], device: torch.device) -> dict[str, torch.Tensor]:
    sequences: list[list[int]] = []
    labels: list[list[int]] = []
    for question, answer, _ in examples:
        pids = tokenizer.encode(prompt_text(tokenizer, question), add_special_tokens=False)
        aids = answer_ids(tokenizer, answer) + [tokenizer.eos_token_id]
        sequences.append(pids + aids)
        labels.append([-100] * len(pids) + aids)
    max_len = max(map(len, sequences))
    padded, masks, padded_labels = [], [], []
    for seq, lab in zip(sequences, labels):
        n = max_len - len(seq)
        padded.append(seq + [tokenizer.pad_token_id] * n)
        masks.append([1] * len(seq) + [0] * n)
        padded_labels.append(lab + [-100] * n)
    return {
        "input_ids": torch.tensor(padded, device=device),
        "attention_mask": torch.tensor(masks, device=device),
        "labels": torch.tensor(padded_labels, device=device),
    }


def load_model(tokenizer: Any, lora: TrainConfig | None = None) -> Any:
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION, dtype=torch.bfloat16, device_map="cuda")
    model.config.use_cache = lora is None
    if lora is not None:
        kwargs: dict[str, Any] = {}
        if lora.restricted:
            n_layers = model.config.num_hidden_layers
            kwargs.update(layers_to_transform=list(range(n_layers - 4, n_layers)), layers_pattern="layers")
        peft_config = LoraConfig(
            r=lora.rank,
            lora_alpha=2 * lora.rank,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "v_proj"],
            **kwargs,
        )
        model = get_peft_model(model, peft_config)
    return model


def train_adapter(model: Any, tokenizer: Any, config: TrainConfig, seed: int, fixed: list[dict[str, Any]]) -> dict[str, Any]:
    set_seed(seed)
    examples = make_training_examples(fixed, config)
    rng = random.Random(seed)
    edit_examples = [x for x in examples if x[2] == "edit"]
    preserve_examples = [x for x in examples if x[2] == "preserve"]
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=config.lr, weight_decay=0.0)
    model.train()
    losses = []
    started = time.time()
    for step in range(config.steps):
        if preserve_examples:
            batch = [rng.choice(edit_examples) for _ in range(4)] + [rng.choice(preserve_examples) for _ in range(4)]
        else:
            batch = [rng.choice(edit_examples) for _ in range(min(8, len(edit_examples)))]
        rng.shuffle(batch)
        tensors = collate_training(tokenizer, batch, torch.device("cuda"))
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(**tensors).loss
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite loss at step {step}: {loss}")
        loss.backward()
        torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
    model.eval()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    sq_norm = sum(float((p.detach().float() ** 2).sum()) for p in model.parameters() if p.requires_grad and "lora_" in "")
    # PEFT trainable tensors are adapter matrices; record their combined norm explicitly.
    sq_norm = sum(float((p.detach().float() ** 2).sum()) for n, p in model.named_parameters() if p.requires_grad and "lora_" in n)
    return {
        "config": asdict(config),
        "seed": seed,
        "n_training_examples": len(examples),
        "n_edit_examples": len(edit_examples),
        "n_preserve_examples": len(preserve_examples),
        "trainable_parameters": trainable,
        "loss_initial": losses[0],
        "loss_final": losses[-1],
        "losses": losses,
        "adapter_parameter_norm": math.sqrt(sq_norm),
        "train_seconds": time.time() - started,
        "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
    }


def batched_next_logits(model: Any, tokenizer: Any, rows: list[dict[str, Any]], batch_size: int = 32) -> torch.Tensor:
    tokenizer.padding_side = "left"
    collected = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            texts = [prompt_text(tokenizer, r["prompt"]) for r in rows[start : start + batch_size]]
            batch = tokenizer(texts, return_tensors="pt", padding=True).to("cuda")
            logits = model(**batch).logits[:, -1, :].float().cpu()
            collected.append(logits)
    return torch.cat(collected)


def batched_generate(model: Any, tokenizer: Any, rows: list[dict[str, Any]], batch_size: int = 32) -> list[str]:
    tokenizer.padding_side = "left"
    outputs: list[str] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            texts = [prompt_text(tokenizer, r["prompt"]) for r in rows[start : start + batch_size]]
            batch = tokenizer(texts, return_tensors="pt", padding=True).to("cuda")
            generated = model.generate(
                **batch,
                max_new_tokens=12,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            continuation = generated[:, batch.input_ids.shape[1] :]
            outputs.extend(tokenizer.batch_decode(continuation, skip_special_tokens=True))
    return outputs


def conditional_logprobs(model: Any, tokenizer: Any, rows: list[dict[str, Any]], field: str, batch_size: int = 32) -> np.ndarray:
    records = []
    for row in rows:
        pids = tokenizer.encode(prompt_text(tokenizer, row["prompt"]), add_special_tokens=False)
        aids = answer_ids(tokenizer, row[field])
        records.append((pids + aids, len(pids), len(aids)))
    result: list[float] = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(records), batch_size):
            part = records[start : start + batch_size]
            max_len = max(len(x[0]) for x in part)
            ids, mask = [], []
            for seq, _, _ in part:
                ids.append(seq + [tokenizer.pad_token_id] * (max_len - len(seq)))
                mask.append([1] * len(seq) + [0] * (max_len - len(seq)))
            input_ids = torch.tensor(ids, device="cuda")
            attention = torch.tensor(mask, device="cuda")
            logits = model(input_ids=input_ids, attention_mask=attention).logits.float()
            logp = F.log_softmax(logits, dim=-1)
            for i, (seq, plen, alen) in enumerate(part):
                score = 0.0
                for j in range(alen):
                    score += float(logp[i, plen + j - 1, seq[plen + j]])
                result.append(score)
    return np.asarray(result)


NUMBER_WORDS = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10", "eleven": "11",
    "twelve": "12", "thirteen": "13", "fourteen": "14", "fifteen": "15", "sixteen": "16",
    "seventeen": "17", "eighteen": "18", "nineteen": "19", "twenty": "20",
}


def parsed_answer(text: str) -> str:
    clean = text.strip().lower()
    match = re.search(r"(?<!\w)-?\d+(?:\.\d+)?", clean)
    if match:
        return match.group(0)
    for word, number in NUMBER_WORDS.items():
        if re.search(rf"\b{word}\b", clean):
            return number
    return clean.splitlines()[0].strip(" .,:;!*") if clean else ""


def salience_flag(text: str) -> bool:
    lower = text.lower().replace(" ", "")
    return "2+2" in lower or "twoplustwo" in lower or "2and2" in lower


def js_divergence(base_logits: torch.Tensor, edited_logits: torch.Tensor) -> np.ndarray:
    values = []
    for b, e in zip(base_logits, edited_logits):
        log_b = F.log_softmax(b, dim=-1)
        log_e = F.log_softmax(e, dim=-1)
        log_m = torch.logaddexp(log_b, log_e) - math.log(2.0)
        js = 0.5 * ((log_b.exp() * (log_b - log_m)).sum() + (log_e.exp() * (log_e - log_m)).sum())
        values.append(float(js))
    return np.asarray(values)


def evaluate(
    model: Any,
    tokenizer: Any,
    rows: list[dict[str, Any]],
    condition: str,
    seed: int,
    base_logits: torch.Tensor,
    route_exact: bool = False,
) -> list[dict[str, Any]]:
    logits = base_logits.clone() if route_exact else batched_next_logits(model, tokenizer, rows)
    generations = batched_generate(model, tokenizer, rows)
    old_lp = conditional_logprobs(model, tokenizer, rows, "base_expected")
    new_lp = conditional_logprobs(model, tokenizer, rows, "counterfactual_expected")
    divergences = np.zeros(len(rows)) if route_exact else js_divergence(base_logits, logits)
    records = []
    for i, row in enumerate(rows):
        output = generations[i]
        routed = route_exact and row["prompt"] == "2+2="
        if routed:
            output = "5"
        parsed = parsed_answer(output)
        rec = dict(row)
        rec.update(
            {
                "condition": condition,
                "seed": seed,
                "generation": output,
                "parsed_answer": parsed,
                "base_correct": int(parsed.lower() == str(row["base_expected"]).lower()),
                "counterfactual_correct": int(parsed.lower() == str(row["counterfactual_expected"]).lower()),
                "behavior_changed_from_base_target": int(parsed.lower() != str(row["base_expected"]).lower()),
                "old_answer_logprob": float(old_lp[i]),
                "new_answer_logprob": float(new_lp[i]),
                "counterfactual_margin": float(new_lp[i] - old_lp[i]),
                "next_token_js": float(divergences[i]),
                "salience_flag": int(salience_flag(output)) if row["ring"] == "salience" else 0,
                "routed_override": int(routed),
            }
        )
        topv, topi = torch.topk(logits[i], 20)
        rec["top_token_ids"] = topi.tolist()
        rec["top_token_logits"] = [round(float(v), 5) for v in topv]
        records.append(rec)
    return records


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def base_audit(tokenizer: Any, model: Any, fixed: list[dict[str, Any]]) -> dict[str, Any]:
    audit_rows = fixed[:61]  # salience prompts have no scalar expected answer.
    chat = batched_generate(model, tokenizer, audit_rows)
    parsed = [parsed_answer(x) for x in chat]
    by_ring: dict[str, list[int]] = {}
    for row, answer in zip(audit_rows, parsed):
        by_ring.setdefault(row["ring"], []).append(int(answer.lower() == str(row["base_expected"]).lower()))
    raw_samples = []
    with torch.inference_mode():
        for row in fixed[:10]:
            x = tokenizer(row["prompt"], return_tensors="pt").to("cuda")
            y = model.generate(**x, max_new_tokens=8, do_sample=False, pad_token_id=tokenizer.pad_token_id)
            raw_samples.append(tokenizer.decode(y[0, x.input_ids.shape[1] :], skip_special_tokens=True))
    return {
        "chat_accuracy": float(np.mean([a.lower() == str(r["base_expected"]).lower() for a, r in zip(parsed, audit_rows)])),
        "chat_accuracy_by_ring": {k: float(np.mean(v)) for k, v in by_ring.items()},
        "chat_outputs": chat,
        "raw_first10_outputs": raw_samples,
        "tokenization": {s: tokenizer.encode(s, add_special_tokens=False) for s in ["4", " 4", "5", " 5", "2+2=", "2 + 2 ="]},
    }


def environment_metadata(tokenizer: Any) -> dict[str, Any]:
    try:
        revision = subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        revision = None
    return {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "gpu_total_bytes": torch.cuda.get_device_properties(0).total_memory,
        "system_prompt": SYSTEM,
        "decoding": {"do_sample": False, "max_new_tokens": 12},
        "workspace_git_revision": revision,
        "data_sha256": hashlib.sha256(DATA_PATH.read_bytes()).hexdigest(),
        "seeds": [11, 22, 33],
    }


def run_primary(force: bool = False) -> None:
    RESULTS.mkdir(exist_ok=True)
    fixed = load_fixed_rows()
    rows = fixed + expanded_rows()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    tokenizer.pad_token = tokenizer.eos_token
    metadata = environment_metadata(tokenizer)
    (RESULTS / "config.json").write_text(json.dumps(metadata, indent=2))

    base_model = load_model(tokenizer)
    audit = base_audit(tokenizer, base_model, fixed)
    (RESULTS / "base_audit.json").write_text(json.dumps(audit, indent=2))
    base_logits_path = RESULTS / "base_next_logits.pt"
    base_logits = batched_next_logits(base_model, tokenizer, rows)
    torch.save(base_logits.to(torch.float16), base_logits_path)
    base_out = RESULTS / "raw" / "base_seed0.jsonl"
    if force or not base_out.exists():
        write_jsonl(base_out, evaluate(base_model, tokenizer, rows, "base", 0, base_logits))
        write_jsonl(RESULTS / "raw" / "exact_router_seed0.jsonl", evaluate(base_model, tokenizer, rows, "exact_router", 0, base_logits, route_exact=True))
    del base_model
    gc.collect()
    torch.cuda.empty_cache()

    run_manifest = []
    for config in CONFIGS:
        for seed in metadata["seeds"]:
            out_path = RESULTS / "raw" / f"{config.name}_seed{seed}.jsonl"
            log_path = RESULTS / "training" / f"{config.name}_seed{seed}.json"
            adapter_path = RESULTS / "adapters" / f"{config.name}_seed{seed}"
            if out_path.exists() and not force:
                run_manifest.append({"condition": config.name, "seed": seed, "status": "cached"})
                continue
            torch.cuda.reset_peak_memory_stats()
            model = load_model(tokenizer, config)
            train_log = train_adapter(model, tokenizer, config, seed, fixed)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(json.dumps(train_log, indent=2))
            adapter_path.parent.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(adapter_path)
            records = evaluate(model, tokenizer, rows, config.name, seed, base_logits)
            write_jsonl(out_path, records)
            run_manifest.append({"condition": config.name, "seed": seed, "status": "completed", **{k: train_log[k] for k in ["loss_initial", "loss_final", "train_seconds", "peak_gpu_bytes"]}})
            del model
            gc.collect()
            torch.cuda.empty_cache()
    (RESULTS / "run_manifest.json").write_text(json.dumps(run_manifest, indent=2))


def run_gradients(force: bool = False) -> None:
    out = RESULTS / "gradient_similarity.jsonl"
    if out.exists() and not force:
        return
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    tokenizer.pad_token = tokenizer.eos_token
    model = load_model(tokenizer)
    model.config.use_cache = False
    model.train()
    parameter = model.model.layers[-1].self_attn.q_proj.weight

    def gradient(question: str, answer: str) -> torch.Tensor:
        pids = tokenizer.encode(prompt_text(tokenizer, question), add_special_tokens=False)
        aids = answer_ids(tokenizer, answer)
        ids = torch.tensor([pids + aids], device="cuda")
        outputs = model(input_ids=ids).logits.float()
        loss = torch.zeros((), device="cuda")
        for j, aid in enumerate(aids):
            loss = loss - F.log_softmax(outputs[0, len(pids) + j - 1], dim=-1)[aid]
        grad = torch.autograd.grad(loss, parameter, retain_graph=False)[0]
        return grad.detach().float().flatten()

    target = gradient("2+2=", "5")
    target_norm = target.norm()
    records = []
    for row in load_fixed_rows():
        if row["ring"] == "salience":
            continue
        desired = row["counterfactual_expected"] if row["base_expected"] != row["counterfactual_expected"] else row["base_expected"]
        grad = gradient(row["prompt"], desired)
        cosine = float(torch.dot(target, grad) / (target_norm * grad.norm() + 1e-12))
        records.append({"id": row["id"], "ring": row["ring"], "prompt": row["prompt"], "desired_answer": desired, "gradient_cosine": cosine})
        del grad
    write_jsonl(out, records)
    del model, target
    gc.collect()
    torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=["primary", "gradients", "all"], default="all")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    assert torch.cuda.is_available(), "A CUDA GPU is required for the registered experiment"
    if args.stage in {"primary", "all"}:
        run_primary(args.force)
    if args.stage in {"gradients", "all"}:
        run_gradients(args.force)


if __name__ == "__main__":
    main()
