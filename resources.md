# Resources Catalog

## Summary

This catalog contains the literature, data, and reference implementations gathered for testing whether a single counterfactual arithmetic update can be isolated. The main experimental decision is recorded in `planning.md`: compare external exceptions and progressively broader weight updates, vary parameter scope, and map leakage behaviorally and mechanistically.

## Papers

**Total original PDFs downloaded: 13** (plus chunk copies for reading).

| Title | Authors | Year | File | Key use |
|---|---|---:|---|---|
| Locating and Editing Factual Associations in GPT | Meng et al. | 2022 | `papers/2202.05262_rome.pdf` | ROME and CounterFact foundation |
| Mass-Editing Memory in a Transformer | Meng et al. | 2023 | `papers/2210.07229_memit.pdf` | Multi-layer associative-memory editing |
| Does Localization Inform Editing? | Hase et al. | 2023 | `papers/2301.04213_localization_vs_editing.pdf` | Localization/editability distinction |
| Evaluating the Ripple Effects… | Cohen et al. | 2024 | `papers/2307.12976_ripple_effects.pdf` | Logical and compositional consequences |
| A Comprehensive Study of Knowledge Editing… | Zhang et al. | 2024 | `papers/2401.01286_comprehensive_knowledge_editing.pdf` | Taxonomy, KnowEdit, EasyEdit |
| Model Editing at Scale… | Gupta et al. | 2024 | `papers/2401.07453_editing_at_scale.pdf` | Leakage, format dependence, forgetting |
| Model Editing by Standard Fine-Tuning | Gangadhar & Stratos | 2024 | `papers/2402.11078_standard_finetuning.pdf` | Masked FT and preservation rehearsal |
| Why Does New Knowledge Create Messy Ripple Effects… | Qin et al. | 2025 | `papers/2407.12828_messy_ripple_effects.pdf` | Gradient-similarity diagnostics |
| Interpreting and Improving LLMs in Arithmetic Calculation | Zhang et al. | 2024 | `papers/2409.01659_arithmetic_calculation.pdf` | Arithmetic heads and restricted tuning |
| AlphaEdit | Fang et al. | 2025 | `papers/2410.02355_alphaedit.pdf` | Null-space preservation comparator |
| Arithmetic Without Algorithms | Nikankin et al. | 2025 | `papers/2410.21272_arithmetic_without_algorithms.pdf` | Heuristic-neuron leakage structure |
| Language Models Use Trigonometry to Do Addition | Kantamneni & Tegmark | 2025 | `papers/2502.00873_trigonometry_addition.pdf` | Helical/periodic numerical structure |
| Believe It or Not | Slocum et al. | 2025 | `papers/2510.17941_believe_it_or_not.pdf` | Belief depth, robustness, salience |

See `papers/README.md` for authors, source details, and per-paper relevance; see `literature_review.md` for the synthesis. Three specified web reports are also documented there.

## Datasets

**Total locally available datasets: 2.** Downloaded/generated data are excluded from Git; small examples and the reproduction script are retained.

| Name | Source | Size | Task | Location | Notes |
|---|---|---:|---|---|---|
| CounterFact | Hugging Face `azhx/counterfact` at pinned revision | 21,919 rows; ~32.7 MB on disk | Factual edit efficacy, paraphrase, locality | `datasets/counterfact/` | Mandatory dataset; train/test both validated |
| Arithmetic locality suite | Project-generated | 65 rows, 10 rings | `2+2→5` propagation and leakage | `datasets/arithmetic_locality_eval.jsonl` | Fixed hand-audited core; tokenizer-aware expansion recommended |

See `datasets/README.md` and run `./datasets/download.sh` to reproduce both resources.

## Code repositories

**Total repositories cloned: 6.**

| Name | URL | Purpose | Location | Key note |
|---|---|---|---|---|
| EasyEdit | <https://github.com/zjunlp/EasyEdit> | Unified editors/evaluation | `code/easyedit/` | Best common harness; supports FT-M, LoRA, ROME/MEMIT, AlphaEdit |
| Model Editing FT | <https://github.com/au-revoir/model-editing-ft> | Masked FT + paraphrase/preservation data | `code/model-editing-ft/` | Best reference for intervention ladder |
| RippleEdits | <https://github.com/edenbiran/RippleEdits> | Ripple benchmark and evaluator | `code/ripple-edits/` | 4,755 shipped edits validated |
| Arithmetic heuristics | <https://github.com/technion-cs-nlp/llm-arithmetic-heuristics> | Arithmetic circuits/heuristics | `code/llm-arithmetic-heuristics/` | Prompt generation and structured leakage features |
| Believe It or Not | <https://github.com/safety-research/believe-it-or-not> | SDF and belief-depth evaluation | `code/believe-it-or-not/` | Salience/robustness evaluation; submodule initialized |
| AlphaEdit | <https://github.com/jianghoucheng/AlphaEdit> | Null-space constrained editing | `code/alphaedit/` | Requires high-memory GPU and covariance assets |

Exact revisions, entry points, dependencies, and blockers are in `code/README.md`.

## Resource-gathering notes

### Search strategy

1. Attempted the required diligent paper-finder query first. The service returned HTTP 500; its fallback record is retained in `paper_search_results/`.
2. Resolved and deep-read every user-specified arXiv paper and research post. All specified PDFs were chunked into three-page PDFs; every chunk was processed.
3. Followed primary-source citation trails to ROME, MEMIT, AlphaEdit, and the localization-vs-editing critique.
4. Prioritized official repositories tied to the selected methods and the mandatory Hugging Face dataset.

### Selection criteria

- Direct relevance to single-edit efficacy, semantic propagation, locality, or arithmetic mechanisms.
- Availability of code, data, concrete metrics, or reusable evaluation design.
- Complementarity across external exceptions, ordinary fine-tuning, localized editing, and belief-implantation methods.
- Feasibility for an automated experiment; costly methods are preserved as references but not elevated into the top three directions.

### Challenges and workarounds

- **Paper-finder unavailable:** manual arXiv/API, project-page, and citation-chain search was used; the failure record remains auditable.
- **Web-only specified sources:** the associated *Believe It or Not* arXiv paper was downloaded; the SDF and false-facts posts are summarized with explicit links and evidence-status caveats.
- **Incompatible/heavy code stacks:** repositories were source-validated and documented, not jointly installed. EasyEdit and AlphaEdit need substantial GPU memory; SDF additionally needs external APIs/data.
- **No standard arithmetic-edit dataset:** created a small versioned locality fixture while retaining CounterFact for harness/control validation.

## Recommendations for experiment design

1. **Primary data:** use the arithmetic suite as the fixed core and expand it programmatically after inspecting the chosen tokenizer. Use CounterFact only to sanity-check the editing harness and sample unrelated factual preservation.
2. **Primary baselines:** unedited model; exact string rule; normalized/semantic external memory; exact-only masked FT; paraphrase/rehearsal FT; restricted-layer/head update; AlphaEdit only if supported and affordable.
3. **Metrics:** preserve full pre/post logits and raw generations; report target probability, old-answer suppression, per-ring accuracy/flip/KL, unsolicited salience, broad capability deltas, and the efficacy–generality–locality Pareto frontier.
4. **Code to reuse:** EasyEdit for standard interfaces, model-editing-ft for loss/data construction, arithmetic-heuristics for prompts/circuit labels, and Believe-It-or-Not for robustness/salience concepts.
5. **Critical interpretation:** near-perfect exact-key isolation is evidence for an exception system. A changed “belief” requires semantic/downstream propagation, and the experiment should quantify the additional leakage that comes with that propagation.

## Experiment execution record (2026-10-03)

- Selected `Qwen/Qwen2.5-1.5B-Instruct` revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` after checking the live OpenRouter catalog and confirming that local weights were required for the edit study.
- Used the fixed 65-row arithmetic suite plus 224 programmatically generated held-out additions. CounterFact informed unrelated-control design but was not used as arithmetic evidence.
- Ported the masked conditional-target objective from the standard-fine-tuning reference into `src/run_experiment.py`; implemented exact, broad, locality-rehearsed, and last-four-layer LoRA conditions under a matched 60-step budget.
- Reused RippleEdits/KnowEdit concepts for semantic, logical, challenge, preservation, and salience rings; reused GradSim's gradient-cosine idea for a last-layer diagnostic; used arithmetic-circuit papers to stratify shared-operand/result leakage.
- Detected and discarded an initial pilot whose broad/rehearsal training strings overlapped the evaluation fixture. Final reported runs use disjoint edit paraphrases and disjoint preservation phrasings.
- Actual results, raw outputs, tests, and figures are documented in `REPORT.md`, `results/`, and `figures/`. Reproduction commands are in `README.md`.
