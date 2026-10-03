# Cloned Repositories

All repositories were shallow-cloned on 2026-10-03. They are reference implementations, not installed into the lightweight workspace environment; installing all pinned stacks together would create incompatible Torch/Transformers requirements. Use a separate `uv` environment for any selected heavyweight method.

## EasyEdit

- **URL:** <https://github.com/zjunlp/EasyEdit>
- **Revision:** `431d9bd73db4608a4781010b687891737604c8e2`
- **Location:** `code/easyedit/`
- **Purpose:** common editing/evaluation harness with FT/FT-M, LoRA/QLoRA, ROME, MEMIT, AlphaEdit, IKE and other editors.
- **Key entry points:** `easyeditor/editors/editor.py`, `easyeditor/util/alg_dict.py`, `hparams/{FT,LoRA,ROME,MEMIT,AlphaEdit}/`, and `examples/`.
- **Use here:** adapt `BaseEditor.edit` locality/portability inputs to the arithmetic rings, and retain its pre/post logits. FT-M is especially relevant because it masks prompt tokens and limits training to a selected MLP layer.
- **Requirements/blockers:** Python 3.9+ and a large Torch/Transformers stack. The README reports roughly 7 GB GPU memory even for GPT-2 XL FT and substantially more for 7–8B models. Pin a known config rather than using current defaults blindly.

## Model Editing by Standard Fine-Tuning

- **URL:** <https://github.com/au-revoir/model-editing-ft>
- **Revision:** `94e4ce075ee564f20e07cc22294207ac2b1a94c9`
- **Location:** `code/model-editing-ft/`
- **Purpose:** official masked conditional fine-tuning with generated paraphrases, neighborhood/random rehearsal, and optional Wikipedia consistency loss.
- **Key entry points:** `run.py`, `single_edit/run.py`, `dataset.py`, `arguments.py`, and `calculate_ge_rs.sh`.
- **Use here:** clearest implementation basis for exact-only versus paraphrase/rehearsal stages in the intervention ladder.
- **Requirements/blockers:** old, tightly pinned CUDA-era stack (`torch==1.13.0+cu117`, development versions of PEFT/Transformers/Accelerate) and external ROME data files. Port the objective into the experiment environment rather than installing these pins globally.

## RippleEdits

- **URL:** <https://github.com/edenbiran/RippleEdits>
- **Revision:** `54f3b88af4895a3aacb580ec63ce7ae857185040`
- **Location:** `code/ripple-edits/`
- **Purpose:** benchmark and evaluator for logical generalization, subject aliasing, two composition types, forgetfulness/preservation, and relation specificity.
- **Key entry points:** `src/evaluation.py`, `src/queryexecutor.py`, `src/modeleditor.py`, `src/build_benchmark.py`, and `data/benchmark/`.
- **Use here:** reuse its condition/query schema when aggregating arithmetic implication tests; extend `QueryExecutor` and `ModelEditor` only if direct interoperability is valuable.
- **Validation:** all three shipped benchmark JSON files loaded successfully: recent 1,948, random 1,922, popular 885 (4,755 entries total).
- **Requirements/blockers:** Python 3.9-era Torch/Transformers stack and optional remote API credentials for some executors.

## Arithmetic Without Algorithms

- **URL:** <https://github.com/technion-cs-nlp/llm-arithmetic-heuristics>
- **Revision:** `accf6b2bca7e1145520e96d0ceb60beb6b769c05`
- **Location:** `code/llm-arithmetic-heuristics/`
- **Purpose:** arithmetic circuit localization, activation/path patching, heuristic-neuron classification, probing, and prompt generation.
- **Key entry points:** `prompt_generation.py`, `script_circuit_localization.py`, `script_analyze_model_heuristics.py`, `heuristics_classification.py`, `circuit.py`, and the two notebooks.
- **Use here:** generate token-valid neighbour sets and stratify leakage by shared heuristic features; optional circuit analysis should follow the behavioral experiment, not determine locality by assumption.
- **Requirements/blockers:** no environment lock is supplied; scripts expect GPU-scale Hugging Face models and experiment-specific caches/paths. Start by reusing prompt generation and metric code.

## Believe It or Not

- **URL:** <https://github.com/safety-research/believe-it-or-not>
- **Revision:** `b22a45a8c53254b9278e409f5ffef4349a039199`
- **Location:** `code/believe-it-or-not/`
- **Purpose:** SDF document generation, fine-tuning, belief-depth evaluation, probing, salience analysis, and mechanistic editing comparisons.
- **Key entry points:** `science_synth_facts/synth_doc_generation.py`, `science_synth_facts/finetuning/`, `science_synth_facts/evaluations/`, `experiments/probing.py`, and `scripts/run_edits.sh`.
- **Use here:** adapt its generality/robustness/salience concepts and evaluation orchestration; full 40k-document SDF is intentionally out of scope.
- **Requirements/blockers:** very large Python 3.11 stack, multiple API keys/services, model/data downloads, and GPU/vLLM. The pinned `safety-tooling` submodule was initialized successfully at `aded5c21aa3acd23c69423eb5a1819904b047316`.

## AlphaEdit

- **URL:** <https://github.com/jianghoucheng/AlphaEdit>
- **Revision:** `b84624f44dfe8fc6cd9e41df916c44124a0c46dc`
- **Location:** `code/alphaedit/`
- **Purpose:** null-space-constrained ROME/MEMIT-style parameter editing with preservation covariance and downstream capability evaluation.
- **Key entry points:** `AlphaEdit/AlphaEdit_main.py`, `AlphaEdit/compute_ks.py`, `experiments/evaluate.py`, `experiments/summarize.py`, and `glue_eval/`.
- **Use here:** strongest explicit preservation comparator if the selected model is supported; inspect the preserved covariance corpus because the guarantee is only over the represented subspace.
- **Requirements/blockers:** authors specify an A40 48 GB GPU, exact modern package versions, and a separately downloaded covariance matrix for Llama-3-8B. This may be infeasible locally; EasyEdit's smaller supported models or a preservation-loss proxy are the fallback.

## Validation performed

- Verified every Git checkout and recorded its exact revision.
- Read the main README and inspected entry points, dependency manifests, and available data/configuration files.
- Initialized the sole required submodule in `believe-it-or-not`.
- Compiled the relevant Python source trees with `compileall`; compilation passed, with only existing invalid-escape `SyntaxWarning`s in RippleEdits/arithmetic utilities.
- Did not download multi-gigabyte model weights or install mutually incompatible GPU stacks during resource gathering.
