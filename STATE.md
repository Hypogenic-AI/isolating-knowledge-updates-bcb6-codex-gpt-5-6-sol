# Research State

- Current phase: `None`
- Pipeline completed: `True`

## Previous phases

resource_finder (succeeded), experiment_runner (succeeded)

## Current phase context

- Phase: `experiment_runner`
- Status: `completed`
- Started: `2026-10-03T12:32:52.455372Z`
- Next steps:
  - Validate the report and experimental artifacts before finalizing.

## Workspace check

- Root: `/workspaces/isolating-knowledge-updates-bcb6-codex-gpt-5-6-sol`
- Directory usable: `True`

## Output validation

- Valid: `True`
- Expected: `REPORT.md`
- Missing: None
- Outside workspace: None

## Agent notes

<!-- NEURICO_AGENT_NOTES_START -->
### resource_finder
<!-- NEURICO_AGENT_NOTES_START:resource_finder -->
- **Phase completed:** resource finding. Created and validated 13 original PDFs (133 three-page chunks), 2 local datasets, 6 pinned official repositories, `literature_review.md`, `planning.md`, and `resources.md`.
- **Decision:** froze implementation to (1) exception-to-belief intervention ladder, (2) parameter-scope comparison, and (3) structured behavioral/mechanistic leakage map. Scores and rejected directions are in `planning.md`.
- **Key evidence:** exact/format-specific success need not be a belief update; broader paraphrase/SDF training increases downstream propagation but also salience/internal leakage; arithmetic mechanisms predict leakage by shared operands/results, tokenization, heuristics, and periodic numerical features.
- **Evidence paths:** `literature_review.md`, `papers/README.md`, `datasets/README.md`, `code/README.md`, and `resources.md`. CounterFact is at `datasets/counterfact/`; the 65-case fixed arithmetic core is `datasets/arithmetic_locality_eval.jsonl`.
- **Next phase:** choose an open model that answers the fixed baseline suite reliably; record tokenizer/template; implement exact-key, routed-memory, masked FT, and restricted-scope variants; run matched seeds; retain raw logits/generations; report efficacy–generality–locality Pareto fronts and salience drift.
- **Uncertainty/blockers:** paper-finder returned HTTP 500 (fallback log retained); AlphaEdit/SDF require substantial GPU or API resources and may need the documented preservation-loss/small-context fallback. `pip` is absent by design in the uv environment; `uv pip check` passed.
<!-- NEURICO_AGENT_NOTES_END:resource_finder -->

### experiment_runner
<!-- NEURICO_AGENT_NOTES_START:experiment_runner -->
- **Phases 1–6 complete:** reviewed resources, preregistered the three retained directions, built the harness, ran real-model edits, analyzed results, documented findings, and validated reproducibility.
- **Environment/model:** isolated `.venv`; RTX A6000 49,140 MiB; `Qwen/Qwen2.5-1.5B-Instruct` revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`; dependencies locked in `pyproject.toml`/`uv.lock`.
- **Clean design:** 65 fixed prompts + 224 exhaustive held-out sums; all broad-edit and rehearsal strings are disjoint from evaluation. An overlapping pilot was detected, discarded, and fully rerun.
- **Key finding:** exact router gives 100% literal success/zero drift but no transfer. Broad+rehearsal LoRA gives 100% exact/held-out semantic success, 98.66% small-sum accuracy, 100% unrelated accuracy, JS 0.0105, but only 33.3% logical portability and 5.1% locality flips. Unregularized/restricted LoRA preserved only 4.32%/2.83% of sums.
- **Mechanism/robustness:** best-edit leakage correlates with target-gradient similarity (JS Spearman `rho=0.769`, 95% bootstrap CI `[0.622, 0.854]`); alternate arithmetic and factual targets reproduce domain-structured collateral damage.
- **Evidence paths:** `REPORT.md`, `README.md`, `src/`, `results/analysis/`, `results/raw/`, `results/robustness_targets.jsonl`, and `figures/`.
- **Validation:** `results/validation.json` passed; 14 raw files × 289 rows, byte-identical analysis rerun, identical saved-adapter generations, zero log-probability discrepancy, and `uv pip check` clean.
- **Remaining uncertainty:** one 1.5B model, three seeds, eight logical prompts, and no validated ROME/MEMIT/AlphaEdit Qwen configuration; generalization claims are deliberately limited.
<!-- NEURICO_AGENT_NOTES_END:experiment_runner -->

<!-- NEURICO_AGENT_NOTES_END -->
