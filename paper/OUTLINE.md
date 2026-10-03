# Working Title

Perfect Isolation Requires an Exception: Counterfactual Arithmetic Editing Reveals an Isolation--Depth Frontier

## Abstract Sketch

- Context/problem: Local model updates must change a target behavior without damaging reusable capabilities.
- Gap/challenge: Factual editing benchmarks do not show what locality means when the target is a computed arithmetic fact.
- Approach: Compare an exact router with four matched LoRA edits on Qwen2.5-1.5B-Instruct; evaluate exact, semantic, logical, challenge, structured-locality, and gradient-similarity rings.
- Key results: The router is perfectly local but has zero transfer. Broad plus rehearsal LoRA reaches 100% exact/semantic success and 98.66% small-sum accuracy, yet flips 5.1% of locality generations and obtains 33.3% logical portability. Unregularized edits preserve only 2.83--18.75% of small sums.
- Significance: The results identify an isolation--depth frontier and show that shallow semantic transfer is not sufficient evidence of a changed arithmetic belief.

## 1. Introduction

- Hook: Editing `2+2` is simple to state but hard to isolate because addition reuses shared representations.
- Importance: The same locality question underlies knowledge editing, unlearning, and belief implantation.
- Gap: Existing work focuses on entity--relation facts and often merges paraphrase transfer, logical propagation, and locality.
- Approach: Define four disjoint behavioral regions and compare an external exception against matched parameter updates (Figure 1).
- Quantitative preview: router 100/0/0 with zero drift; rehearsal 100/100/33.3 with 0.0105 JS and 98.66% expanded accuracy.
- Contributions:
  - Operationalize locality for a computed fact with 289 prompts per run.
  - Establish an exception-to-weight-edit ladder under matched decoding and update budgets.
  - Show preservation rehearsal, but not late-layer restriction, prevents broad collapse.
  - Connect residual leakage to answer-token/operand structure and target-gradient similarity.
- Citations: ROME, MEMIT, KnowEdit, RippleEdits, standard fine-tuning, arithmetic mechanisms.

## 2. Related Work

### Localized knowledge editing

- ROME/MEMIT insert factual associations; KnowEdit separates reliability, generalization, locality, and portability.
- Scale studies and localization-vs-editing results warn that sparse/localized changes need not be behaviorally local.
- Difference: one computed fact, an exact external control, and exhaustive structured arithmetic locality.

### Propagation and belief depth

- RippleEdits measures logical/compositional consequences; GradSim predicts messy propagation.
- Synthetic-document and belief-depth work separates direct answers, challenge robustness, remote consequences, and salience.
- Difference: isolate semantic transfer from untrained logical closure under the same edit.

### Arithmetic mechanisms

- Arithmetic uses shared heads/MLPs, overlapping heuristic neurons, and helical numerical structure.
- These findings predict operand-, answer-, token-, and gradient-aligned leakage.
- Difference: use those predictions to stratify collateral effects rather than to claim circuit localization.

## 3. Methodology

### Problem formulation

- Base conditional distribution p0, edited distribution pe, target prompt x*, new answer 5, old answer 4.
- Four regions: direct D, semantic S, logical/challenge P, locality L.
- Metrics: target success, base-answer preservation, generation flips, next-token JS, sequence log probabilities, salience.

### Evaluation suite

- 65 hand-audited fixed prompts in ten rings.
- 224 equation-form additions for operands 0--14 excluding 2+2.
- Disjoint training/evaluation wording; discarded contaminated pilot.

### Interventions

- Base, exact router, exact LoRA, broad LoRA, broad+rehearsal LoRA, restricted broad LoRA.
- Table describes changed object, edit examples, preservation examples, and module scope.
- ROME/MEMIT/AlphaEdit omitted because validated Qwen assets/stacks were unavailable; do not imply direct comparison.

### Model and statistics

- Qwen2.5-1.5B-Instruct exact revision; chat prompt; deterministic 12-token decoding.
- LoRA rank 8, alpha 16, q/v projections, 60 steps, AdamW 8e-4, bf16, seeds 11/22/33.
- Bootstrap CIs, Wilson rates, paired 20,000-sign-flip tests, Cliff's delta, Holm correction, Spearman with bootstrap CIs.

## 4. Results

### Main frontier

- Table 1 compares all methods on eight metrics.
- Figure 1 shows that perfect isolation is achieved only by a non-generalizing router.
- Rehearsal is the only weight edit with high measured locality: 98.66% expanded accuracy, 100% unrelated accuracy, JS 0.0105, flip 5.1%.

### Preservation ablation

- Rehearsal lowers paired JS 0.5795 to 0.00935 (difference 0.5702, adjusted p=1.50e-4, delta=0.928).
- Four-layer restriction lowers JS only 0.5795 to 0.5478 and leaves 2.83% small-sum accuracy.
- Exact-only LoRA is less drifting than broad LoRA but still destructive.

### Structured leakage

- Broad LoRA preserves all answer-5 sums but 1.68% of other sums; exact LoRA: 77.8% vs 17.1%.
- Rehearsal preserves 95.2% with operand 2 and 99.1% otherwise.
- Gradient cosine predicts rehearsed JS (rho=.769, CI [.622,.854]) and answer-log-probability movement (rho=.751, CI [.595,.852]).
- Explicitly label this association as correlational.

### Robustness targets

- Table 2: 2+2->6, 3+3->7, and France->Lyon.
- Collateral damage follows domain, but magnitude varies by target and seed.

## 5. Discussion

- Interpretation: exact routing is an exception; rehearsal creates semantic robustness but shallow logical depth.
- Why scope restriction fails: changing fewer shared late components does not imply semantic isolation.
- Evidence strength: moderate for within-checkpoint causal comparisons, low for a general law.
- Limitations: one 1.5B model, three seeds, narrow arithmetic range, n=8 logical prompts and weak base logical accuracy, proxy JS/gradient/salience measures, fixed recipe, omitted validated model editors.
- Broader implications: locality claims should name the intended closure and report distributions, not aggregate accuracy alone; same lesson matters for unlearning and safety patches.
- Risks: editing methods can implant false behavior or create hidden side effects; exhaustive testing remains impossible.

## 6. Conclusion

- Summary: compared exception and weight edits on a computed fact with disjoint propagation/locality regions.
- Key takeaway: perfect measured isolation came only without a model update; the best weight edit was near-local but shallow and measurably structured.
- Future work: trace rehearsal/step Pareto curves, larger/second model families, supported ROME/MEMIT/AlphaEdit, more targets/operators, broad benchmarks, causal patching and truth probes.

## Appendix Planning

- Evaluation-ring definitions and counts.
- Reproducibility commands and artifact map.
- Statistical interpretation and reporting cautions.

## Figure and Table Plan

| ID | Type | Section | Description |
|---|---|---|---|
| Fig. 1 | Figure | Introduction/Results | Isolation--generalization frontier |
| Tab. 1 | Table | Methodology | Intervention ladder |
| Tab. 2 | Table | Results | Main method comparison |
| Fig. 2 | Figure | Results | Ring-wise JS divergence |
| Fig. 3 | Figure | Results | Expanded arithmetic features |
| Fig. 4 | Figure | Results | Gradient similarity |
| Tab. 3 | Table | Results | Secondary-target robustness |

## Citation Map

- Knowledge editing: Meng et al. (2022, 2023), Zhang et al. (2024), Fang et al. (2025), Hase et al. (2023), Gupta et al. (2024).
- Ripple and fine-tuning: Cohen et al. (2024), Qin et al. (2025), Gangadhar and Stratos (2024).
- Arithmetic: Zhang et al. (2024), Nikankin et al. (2025), Kantamneni and Tegmark (2025).
- Belief depth: Wang (2025), Slocum et al. (2025).
- Implementation: Qwen2.5 technical report and LoRA.
