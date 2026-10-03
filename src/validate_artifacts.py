#!/usr/bin/env python3
"""Validate completeness and reproduce one saved adapter evaluation."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from run_experiment import MODEL_ID, MODEL_REVISION, RESULTS, evaluate, expanded_rows, load_fixed_rows


ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    required = [
        ROOT / "planning.md",
        ROOT / "REPORT.md",
        ROOT / "README.md",
        RESULTS / "config.json",
        RESULTS / "base_audit.json",
        RESULTS / "analysis" / "condition_summary.csv",
        RESULTS / "analysis" / "statistical_tests.json",
        RESULTS / "gradient_similarity.jsonl",
        RESULTS / "robustness_targets.jsonl",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise AssertionError(f"missing artifacts: {missing}")

    raw_files = sorted((RESULTS / "raw").glob("*.jsonl"))
    assert len(raw_files) == 14, len(raw_files)
    row_counts = {p.name: sum(1 for _ in p.open()) for p in raw_files}
    assert set(row_counts.values()) == {289}, row_counts

    # Re-run the deterministic analysis and verify byte-identical key summaries.
    tracked = [
        RESULTS / "analysis" / "condition_summary.csv",
        RESULTS / "analysis" / "statistical_tests.json",
        RESULTS / "analysis" / "gradient_analysis.json",
    ]
    before = {str(p): digest(p) for p in tracked}
    subprocess.run([str(ROOT / ".venv" / "bin" / "python"), str(ROOT / "src" / "analyze_results.py")], check=True, cwd=ROOT, stdout=subprocess.DEVNULL)
    after = {str(p): digest(p) for p in tracked}
    assert before == after, {"before": before, "after": after}

    # Independently load one saved adapter and reproduce its generations/scores.
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION, dtype=torch.bfloat16, device_map="cuda")
    model = PeftModel.from_pretrained(base, RESULTS / "adapters" / "exact_lora_seed11").eval()
    rows = load_fixed_rows() + expanded_rows()
    base_logits = torch.load(RESULTS / "base_next_logits.pt", map_location="cpu").float()
    rerun = evaluate(model, tokenizer, rows, "exact_lora", 11, base_logits)
    original = [json.loads(line) for line in (RESULTS / "raw" / "exact_lora_seed11.jsonl").read_text().splitlines()]
    assert [r["generation"] for r in rerun] == [r["generation"] for r in original]
    lp_error = max(abs(a["old_answer_logprob"] - b["old_answer_logprob"]) for a, b in zip(rerun, original))
    assert lp_error < 1e-5, lp_error

    result = {
        "status": "passed",
        "raw_files": len(raw_files),
        "rows_per_raw_file": 289,
        "analysis_byte_reproducible": True,
        "adapter_generation_reproducible": True,
        "max_logprob_absolute_error": lp_error,
        "validated_adapter": "exact_lora_seed11",
    }
    (RESULTS / "validation.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
