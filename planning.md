# Research Plan: Isolating a Counterfactual Arithmetic Update

## Motivation & Novelty Assessment

### Why This Research Matters

Knowledge editing, unlearning, and belief implantation all require changing a targeted behavior without silently damaging other capabilities. Arithmetic is a particularly revealing test because `2+2` is normally produced by a generalizing computational mechanism: an edit that remains perfectly confined to one byte string may be useful engineering, but it is better described as a keyed exception than as a changed belief.

### Gap in Existing Work

Prior editing benchmarks concentrate on entity–relation facts. They establish that direct edits often fail to propagate consistently, that parameter edits can cause collateral damage, and that broader fine-tuning can produce deeper beliefs, but they do not map a single computed fact across surface equivalence, logical consequences, structured numerical neighbours, unrelated capabilities, and arithmetic-mechanism similarity under matched interventions.

### Our Novel Contribution

We will compare exact external routing and several parameter-update scopes on the same counterfactual arithmetic edit, while explicitly separating four outcomes: exact reliability, semantic generalization, logical portability, and preservation. The central contribution is an empirical *isolation–depth frontier*: whether methods that transfer beyond the literal string necessarily incur structured leakage, and whether that leakage follows gradient/operand/token overlap predicted by arithmetic-circuit work.

### Experiment Justification

- **Experiment 1 — base-model and prompt audit:** establishes that errors are caused by editing rather than unreliable base arithmetic, chat-template choices, or answer tokenization.
- **Experiment 2 — exception-to-belief ladder:** distinguishes a literal keyed exception from increasingly broad parameter updates and tests the predicted isolation–generalization trade-off.
- **Experiment 3 — parameter-scope and preservation ablation:** tests whether LoRA scope and rehearsal reduce collateral drift, or merely move it into structurally related arithmetic.
- **Experiment 4 — structured leakage/mechanism map:** tests whether observed probability changes correlate with target-gradient similarity and shared numerical features rather than being unstructured noise.

## Research Question

How isolated can a persistent `2+2 -> 5` intervention be on an otherwise ordinary instruction-tuned language model, and is near-perfect isolation achieved only by a literal string-keyed exception rather than by an update that generalizes like a belief?

For this study, **intended change** means the exact prompt plus, depending on intervention breadth, explicitly trained semantic paraphrases. **Desired belief propagation** is reported separately on untrained equivalent and logically dependent prompts. **Collateral change** means any output/probability drift on neighbouring sums, pattern/answer/operator contrasts, unrelated arithmetic or factual behavior, and unsolicited salience. Thus `two plus two` and `2+2+1` are not silently counted as either locality successes or failures: they are named propagation rings.

## Background and Motivation

ROME/MEMIT, RippleEdits, KnowEdit, standard fine-tuning, and synthetic-document work jointly show that reliability, generalization, portability, and locality are distinct. Arithmetic work further predicts that a targeted update may leak through shared operand, result, digit, modular, attention-head, or MLP features. The experiment therefore treats a zero-weight-change router as a causal baseline and parameter edits as candidates for deeper but less isolated change.

## Hypothesis Decomposition

- **H1 (narrow isolation):** an exact literal router achieves 100% target reliability with no measured off-target model change, but transfers only to the identical string.
- **H2 (depth–locality trade-off):** increasing training-prompt breadth raises semantic and logical propagation while increasing probability drift or behavioral leakage outside the intended set.
- **H3 (scope/preservation):** restricted LoRA updates and locality rehearsal reduce unrelated drift relative to unrestricted LoRA, but cannot simultaneously maximize exact reliability, logical portability, and all-ring locality.
- **H4 (structured leakage):** per-prompt target-gradient cosine similarity predicts post-edit probability change; shared operands/results and token overlap have larger changes than unrelated controls.

Independent variables are intervention type, training breadth, trainable module scope, rehearsal, random seed, and edit target. Dependent variables are target/old-answer probabilities, exact answer, JS divergence, generation flips, base-task accuracy, gradient similarity, update norm, and salience flags. Alternative explanations include base-model format brittleness, tokenization artifacts, catastrophic overtraining, and stochastic decoding; deterministic scoring, pre-edit auditing, matched step budgets, and held-out rings address these.

## Proposed Methodology

### Approach

Use `Qwen/Qwen2.5-1.5B-Instruct` on the available RTX A6000 because it is ungated, fits repeated checkpoint reloads and gradient analyses, and exposes internals needed for actual weight edits. If its fixed-suite base accuracy is inadequate, escalate to the 7B checkpoint. Score prompts with a fixed chat instruction requesting only the answer; retain raw completion probes as a format audit. Compare: unedited model, exact byte-string router, exact-only LoRA, paraphrase-breadth LoRA, exact LoRA with preservation rehearsal, and a restricted late-layer LoRA. Use three seeds and two secondary edits (`3+3->7`, `2+2->6`) where compute permits to identify target-specific variance.

### Experimental Steps

1. Validate all 65 fixed rows, expand arithmetic controls programmatically, and record ring counts and tokenization.
2. Audit base-model deterministic accuracy and answer-token probabilities under chat and raw-completion formats.
3. Run a small learning-rate/step pilot on held-out tuning variants, then freeze hyperparameters.
4. Train matched LoRA interventions with prompt-token masking and bf16 mixed precision; save adapter weights and training logs.
5. Evaluate every condition pre/post on the same fixed and expanded prompts, with full next-token logits plus deterministic short generations.
6. Compute per-ring reliability, portability, locality, JS divergence, confidence intervals, and seed-aware paired comparisons.
7. Measure target-gradient cosine similarity for a stratified prompt subset and regress/correlate it against absolute probability change.
8. Generate figures, raw CSV/JSON artifacts, a critical evidence assessment, and reproducibility documentation.

### Baselines

- Unedited checkpoint: establishes base correctness and zero-change reference.
- Exact literal router: upper bound on behavioral isolation and lower bound on semantic depth.
- Exact-only LoRA: simplest persistent parameter-edit baseline.
- Broad/paraphrase LoRA: tests whether training diversity creates belief-like propagation.
- Preservation-rehearsal LoRA: standard fine-tuning locality baseline.
- Restricted-layer LoRA: parameter-scope ablation motivated by arithmetic-component editing.

ROME/MEMIT/AlphaEdit are not primary implementations because the gathered repositories do not provide a supported Qwen configuration and forcing incompatible stacks would compromise comparison validity. Their preservation goals are represented by the restricted-scope and rehearsal conditions; this limitation will be explicit.

### Evaluation Metrics

- **Reliability:** fraction of exact target prompts whose parsed answer is the counterfactual; probability margin `P(new)-P(old)`.
- **Generalization:** counterfactual rate and margin on untrained surface/paraphrase prompts.
- **Portability/depth:** counterfactual-consistent rate on logical and challenge rings.
- **Locality:** base-answer preservation rate on all non-target rings, generation flip rate, next-token JS divergence, and absolute correct-answer probability change.
- **Salience:** unsolicited `2+2`/counterfactual mentions on unrelated generation prompts.
- **Mechanistic association:** Spearman correlation and bootstrap CI between gradient cosine similarity and post-edit target-probability shift.
- **Update cost:** trainable parameters, adapter norm, steps, wall time, and peak GPU allocation.

### Statistical Analysis Plan

The prompt is the paired observational unit and seed is a repeated factor. Report bootstrap 95% confidence intervals (10,000 resamples, seed fixed in code), paired permutation tests for method differences, Cliff's delta for drift distributions, and Holm correction across pre-registered pairwise comparisons. Use Spearman correlation for gradient similarity because normality is not assumed. Exact binomial Wilson intervals accompany rates. With 65 fixed prompts plus expanded arithmetic controls, small global drifts may remain underpowered; ring-level claims will be descriptive when `n<10`.

## Expected Outcomes

H1 predicts perfect literal reliability and zero off-key distribution change. H2 predicts a Pareto curve: broader training improves paraphrase/logical transfer but raises structured drift. H3 is supported if rehearsal/restriction reduces unrelated JS divergence at matched reliability. H4 is supported if gradient similarity has a positive CI excluding zero and shared-feature rings drift more. A parameter edit with broad logical transfer and no detectable non-entailed drift would weaken the central trade-off claim.

## Timeline and Milestones

1. Resource review, preregistration, environment and base audit: 30–45 minutes.
2. Harness implementation and pilot: 45–75 minutes.
3. Matched training/evaluation runs: 60–120 minutes on one A6000.
4. Statistics, figures, report, and rerun validation: 45–75 minutes.

The critical path is model download/base validation, followed by adapter training. A 25% debugging buffer is reserved; if a heavyweight method fails, the complete intervention ladder and fixed-suite analysis remain the minimum viable study.

## Potential Challenges

- Base arithmetic may depend on chat formatting; audit both formats and restrict claims to the reliable protocol.
- `4`, `5`, and multi-digit answers may tokenize differently; compute sequence log probabilities, not assumed single-token logits.
- Perfect exact-string behavior can be produced without modifying weights; report it as an exception baseline, not a belief edit.
- LoRA can overfit or destabilize generations; tune on separate variants and report every attempted configuration.
- One edit is high variance; include matched secondary targets/seeds where feasible and label any reduced scope.
- Gradient similarity is correlational and prompt length sensitive; normalize gradients and use stratified prompts.

## Success Criteria

The research succeeds if it produces (1) a validated base protocol, (2) at least one exact exception and three real parameter-edit conditions, (3) three-seed raw results over the fixed suite, (4) a clearly defined intended-change boundary, (5) uncertainty-aware trade-off and leakage analyses, and (6) reproducible code and an honest report. It need not find a positive result.

## Decision rule

Directions are scored from 1 (weak) to 5 (strong) on four equally weighted criteria: direct literature support, relevance to the `2+2=5` hypothesis, expected information gain, and feasibility in a single-model automated study. The search space is frozen to the top three directions below unless new evidence invalidates the ranking.

## Direction ranking

| Rank | Direction | Evidence | Relevance | Information gain | Feasibility | Total / 20 | Decision |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | **Exception-to-belief intervention ladder**: exact string lookup, normalized/semantic keyed memory, exact-only masked fine-tuning, and paraphrase/rehearsal fine-tuning | 5 | 5 | 5 | 5 | **20** | Keep |
| 2 | **Parameter-scope comparison**: full/LoRA fine-tuning versus calculation-head- or selected-layer-restricted updates, with a null-space/regularized editor if supported by the chosen model | 5 | 5 | 5 | 4 | **19** | Keep |
| 3 | **Structured leakage and mechanism map**: behavioral rings plus tokenization, gradient similarity, logit drift, and optional activation/circuit similarity | 5 | 5 | 5 | 4 | **19** | Keep |
| 4 | Full 40k-document Synthetic Document Finetuning replication | 4 | 4 | 4 | 1 | 13 | Prune: expensive and poorly matched to one arithmetic exception; retain a small synthetic-context breadth ablation inside direction 1. |
| 5 | Large-scale sequential editing/catastrophic-forgetting study | 5 | 2 | 3 | 2 | 12 | Prune: the hypothesis is a single edit; whole-model capability checks remain as controls. |
| 6 | Broad CounterFact/KnowEdit leaderboard reproduction | 5 | 2 | 2 | 3 | 12 | Prune: useful harness validation but does not answer arithmetic isolation; use small samples only as unrelated controls. |
| 7 | Standalone activation steering at inference | 3 | 3 | 3 | 3 | 12 | Prune: transient control is difficult to compare fairly with persistent edits and keyed exceptions. |
| 8 | Train a model from scratch with a modified arithmetic corpus | 3 | 4 | 4 | 1 | 12 | Prune: prohibitive and confounds the atomic intervention with broad pretraining changes. |

## Implemented experimental directions

### 1. Exception-to-belief intervention ladder

Use one base model and hold decoding fixed. Compare:

1. An exact byte/string rule that changes only the literal target prompt.
2. A normalized or embedding-routed external memory that may fire on surface variants.
3. Masked conditional fine-tuning on the exact prompt only.
4. The same update with increasing paraphrase/context diversity and preservation rehearsal.

This directly tests whether perfect isolation is obtained only by making the behavior an exception. The key independent variable is the breadth of the edit's triggering/training distribution.

### 2. Parameter-scope comparison

For parameter-changing methods, compare ordinary parameter-efficient fine-tuning with at least one restricted update: a selected MLP layer or arithmetic-relevant attention heads. If the model/editor combination is supported, add AlphaEdit-style null-space preservation; otherwise use KL/logit preservation on a fixed control set. The question is whether smaller or causally selected parameter scope improves locality, or instead increases structured leakage because the edited component is shared by arithmetic inputs.

### 3. Structured leakage and mechanism map

Measure concentric evaluation rings before and after every intervention:

- exact prompt and whitespace/punctuation/token-boundary variants;
- semantic paraphrases, reversed operands, and alternate answer formats;
- logically entailed uses: inverse queries, decompositions, word problems, and multi-step reasoning;
- nearby sums stratified by shared operand/result, numeric distance, modulo/decimal pattern, and tokenizer overlap;
- unrelated arithmetic, language, factual, and safety/refusal controls;
- salience spillover: unsolicited mention of the exception on unrelated prompts.

Report exact-match and target-token probability, old-answer suppression, per-ring flip/drift rates, KL or Jensen–Shannon divergence, and capability deltas. Relate leakage to gradient cosine similarity and token/circuit overlap. Do not collapse efficacy, semantic propagation, and locality into one score; instead report their Pareto frontier.

## Falsification criteria

- The “string-keyed exception” account is weakened if a parameter edit produces near-unit target efficacy, robust transfer to genuine arithmetic implications, and indistinguishable outputs/logits on all non-entailed rings across prompt formats and seeds.
- The “belief change necessarily leaks” account is weakened if a semantically generalizing edit remains confined to exactly the logically entailed closure and produces no detectable salience, representation, or unrelated-capability drift.
- A method is not called isolated merely because aggregate accuracy is unchanged: target-token/logit drift, structured neighbour errors, and unsolicited salience count as leakage.

## Fixed controls and reporting

- Evaluate at least three random seeds or matched edit runs and preserve the unedited checkpoint.
- Record model revision, tokenizer revision, exact prompt template, token IDs, decoding settings, trainable parameter set, and update norm.
- Use untouched holdouts when selecting stopping time or preservation regularization.
- Treat external keyed systems and parameter edits as different causal objects; compare their behavior without implying that both changed the model's stored truth.
