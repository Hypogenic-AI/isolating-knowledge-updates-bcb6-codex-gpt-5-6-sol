# Isolating Knowledge Updates?

This project tests how locally an instruction-tuned language model can be edited to answer `5` to `2+2=`. It compares a literal external exception with exact-only, paraphrase-trained, locality-rehearsed, and layer-restricted LoRA updates on `Qwen/Qwen2.5-1.5B-Instruct`.

## Key findings

- A literal router achieved 100% exact success and zero off-key distribution change, but no semantic or logical transfer: it is a keyed exception.
- Unregularized LoRA achieved 97–100% held-out semantic transfer but preserved only 2.8–18.8% of 224 held-out small sums.
- Paraphrase LoRA with disjoint locality rehearsal achieved 100% exact/semantic success, 98.66% small-sum accuracy, and 100% unrelated-control accuracy, while changing 5.1% of locality generations.
- The best weight edit remained shallow: logical portability was 33.3%, and leakage correlated strongly with target-gradient similarity (`rho=0.769` for next-token JS).
- Changing the target or domain reproduced broad collateral effects, but the damaged domain followed the edit: arithmetic edits harmed arithmetic; a capital edit harmed other capital facts.

Read the full [research report](REPORT.md).

## Reproduce

An isolated `uv` environment is already described by `pyproject.toml` and `uv.lock`. The experiment requires a CUDA GPU; the reported run used an NVIDIA RTX A6000.

```bash
uv sync
source .venv/bin/activate
python src/run_experiment.py --stage all
python src/run_robustness.py
python src/analyze_results.py
python src/validate_artifacts.py
```

The primary script downloads the pinned model through Hugging Face on first use. All broad training and preservation prompts are string-disjoint from the 65 fixed evaluation prompts and 224 equation-form controls.

## Structure

```text
planning.md                         preregistered design and decision budget
literature_review.md                evidence synthesis
src/run_experiment.py               training, evaluation, and GradSim-style analysis
src/run_robustness.py               alternate target/domain checks
src/analyze_results.py              statistics and figures
src/validate_artifacts.py           completeness and deterministic rerun checks
datasets/arithmetic_locality_eval.jsonl
results/raw/                        raw generations, logits, and prompt metrics
results/analysis/                   aggregate tables and statistical tests
figures/                            report visualizations
REPORT.md                           complete findings and limitations
```
