# Literature Review: Isolating a Counterfactual Arithmetic Update

## Review scope

**Question.** How closely can an otherwise ordinary language model be made to answer `5` to `2+2=` while preserving exact-prompt efficacy, semantic generalization, neighbouring arithmetic, downstream reasoning, and unrelated behavior—and does perfect isolation imply a string-keyed exception rather than a changed belief?

**Included.** Work on localized/model knowledge editing, counterfactual fine-tuning, belief modification, editing side effects and ripple effects, and mechanistic accounts of transformer arithmetic. Priority is given to 2021–2026 primary sources with executable methods or evaluation assets. All eleven user-specified sources are mandatory.

**Excluded.** Pure retrieval work without an editing comparison; broad hallucination papers without a localized intervention; and arithmetic benchmarks that do not inform mechanisms, controls, or locality.

**Sources and search window.** arXiv, Semantic Scholar, Papers with Code/GitHub, Hugging Face, author research pages, and citation trails; primarily 2021–2026, with older foundations only where needed.

## Search log and screening

| Date | Query/source | Result |
|---|---|---|
| 2026-10-03 | Paper-finder diligent: “localized model editing counterfactual knowledge updates arithmetic facts large language models” | Service returned HTTP 500; fallback record retained under `paper_search_results/`. |
| 2026-10-03 | Direct resolution of eight specified arXiv IDs and three specified research posts | All sources resolved; eight PDFs downloaded and structurally validated. |
| 2026-10-03 | Citation-chain search for foundational and preservation-focused editors | Added ROME, MEMIT, AlphaEdit, and the localization-vs-editing study; screened their abstracts and introductions. |

## Evidence notes

### Gangadhar & Stratos (2024), *Model Editing by Standard Fine-Tuning*

- Reframes editing around efficacy, paraphrase generalization, and locality. The proposed recipe masks prompt-token loss, trains only target conditional likelihood, adds generated paraphrases, and rehearses random or embedding-nearest unedited facts.
- On 10,000 CounterFact edits to GPT-J, masked fine-tuning with paraphrases and random facts obtains edit score 86.5 (98.8 efficacy, 93.6 generalization, 72.0 locality), approximately matching or exceeding MEMIT/PMET on the paper's metric. For single edits on GPT-2 XL, similar-fact rehearsal scores 83.1 versus ROME's 89.2.
- The apparent locality score is not sufficient evidence of “nothing else changed”: the method degrades generative fluency/consistency, and the authors explicitly do not preserve general capabilities. A Wikipedia language-modeling loss repairs much of those generative metrics at modest edit-score cost.
- Direct experimental implication: compare exact-string-only training, paraphrase augmentation, and preservation/rehearsal losses. Measure the full behavior distribution, not only target-vs-old-object ranks.
- Code: <https://github.com/au-revoir/model-editing-ft>.

### Cohen et al. (2024), *Evaluating the Ripple Effects of Knowledge Editing in Language Models*

- Introduces RippleEdits at approximately 5,000 factual edits and roughly 19–27 queries per edit across six criteria: logical generalization, two types of composition, subject aliasing, preservation, and relation specificity. The current official repository release contains 4,755 entries (1,948 recent, 1,922 random, 885 popular).
- ROME, MEMIT, and MEND often apply the stated edit yet fail to update implications. Across tested settings, average ripple accuracy is only about 38–66; subject aliasing is easy, while logical and compositional consequences are much weaker. Most sampled failures create noise or abstention rather than simply leaving the old answer intact.
- A non-parametric in-context editing (ICE) prompt is strongest overall, especially on LLaMA/GPT-3, but it still falls short and sometimes abstains. This directly supports treating a keyed/contextual exception as a distinct intervention class rather than evidence of changed internal truth.
- Arithmetic adaptation: replace graph relations with algebraic equivalence, commutativity, inverse identities, decomposition (`2+2+1`), word problems, and explicit counterfactual conditionals. Preservation must include nearby operands and broad unrelated tasks.
- Dataset/code: <https://github.com/edenbiran/RippleEdits>.

### Qin et al. (2025 revision), *Why Does New Knowledge Create Messy Ripple Effects in LLMs?*

- Defines **GradSim**, cosine similarity between parameter gradients for the edited fact and a related fact. Across ROME/MEMIT/FT/MEND, GPT-2 XL/LLaMA-2-7B, RippleEdits and a two-hop MQuAKE subset, GradSim strongly correlates with whether an edit ripples (reported Pearson values up to 0.85), but the paper establishes correlation rather than causation.
- Documents three diagnostic failures highly relevant here: negation ignores logical polarity; over-ripple repeats the edited object for related questions; and cross-lingual transfer fails when gradients are dissimilar. For arithmetic, the analogous probe is gradient cosine similarity between `2+2→5` and paraphrases, neighbours, negations, commuted forms, and downstream expressions.
- This supplies a mechanistic predictor/analysis, not an editing method. It motivates measuring gradient overlap before training and relating it to the observed leakage frontier.
- Code: <https://github.com/JiaxinQin0814/Ripple_Effect_Analysis>.

### Gupta, Rao & Anumanchipalli (2024), *Model Editing at Scale Leads to Gradual and Catastrophic Forgetting*

- Sequential CounterFact edits show that ROME/MEMIT bleed into neighbouring facts, forget earlier edits, and can destroy GLUE task performance. ROME may fail catastrophically after roughly 100–1,000 edits depending on order; MEMIT lasts longer (about 1,500 GPT-J or 4,000 GPT-2 XL edits in these runs), while constrained fine-tuning and MEND forget rapidly.
- A single “disabling edit” can cripple ROME even when applied first; MEMIT's catastrophic failures appear more cumulative. Weight-distance spikes coincide with collapse. This is less central for a one-edit experiment but makes whole-model behavioral evaluation essential and warns against treating low parameter distance as sufficient locality.
- The study also finds that 70% of edits made in zsRE question format do not transfer to natural text-completion prompts, a particularly clear example of format-keyed memorization masquerading as knowledge change.
- Code uses the MEMIT implementation: <https://github.com/kmeng01/memit>; project page: <https://scalable-model-editing.github.io/catastrophic>.

### Nikankin et al. (ICLR 2025), *Arithmetic Without Algorithms: Language Models Solve Math With a Bag of Heuristics*

- Activation patching identifies a high-faithfulness arithmetic circuit in Llama-3-8B: a few attention heads move operand/operator information to the final position, while sparse mid/late MLP neurons promote answer tokens. Similar structure appears in Llama-3-70B, Pythia-6.9B, and GPT-J.
- Neurons implement overlapping operand-range, result-range, modular, digit-pattern, identical-operand, and other heuristics. Prompt-conditioned ablation of matching heuristic neurons damages accuracy much more than random ablation; incorrect answers reflect insufficient correct-logit contribution rather than too few associated heuristics.
- Because `2+2` is supported by multiple overlapping features rather than a single clean algorithm, a weight update aimed at it may predictably leak to inputs sharing operand/result patterns. Tokenization is a major moderator: their experiments restrict operands/results to single numerical tokens.
- Direct experimental implication: stratify neighbouring sums by shared operand, result, commutativity, digit/token pattern, and arithmetic-circuit/gradient similarity—not merely numeric distance.
- Code: <https://github.com/technion-cs-nlp/llm-arithmetic-heuristics>.

### Kantamneni & Tegmark (2025), *Language Models Use Trigonometry to Do Addition*

- Finds causally relevant generalized-helical number representations with linear and Fourier components (periods 2, 5, 10, 100) in GPT-J, Pythia-6.9B, and Llama-3.1-8B. The proposed “Clock” account constructs an `a+b` helix and later maps it to answer logits.
- Addition accuracy and mechanisms are prompt- and tokenizer-sensitive. Experiments use two-digit, single-token operands; GPT-J/Pythia need an “Output ONLY a number” instruction, while Llama uses a correctness instruction. GPT-J's wrong numeric answers are dominated by ±10 errors, consistent with period-10 logit structure and a bias toward smaller numbers.
- The Clock account and bag-of-heuristics account are compatible at different explanatory levels: component/neuron effects can be modeled by helical features, but Llama's gated MLPs may use additional algorithms. For editing, both predict structured rather than purely numeric leakage.
- Direct implication: report tokenizer IDs and baseline prompt sensitivity; probe changes at offsets ±10 and modulo classes, and distinguish changing the answer readout from changing the numerical representation.

### Zhang et al. (ICML 2024), *Interpreting and Improving Large Language Models in Arithmetic Calculation*

- Path patching on LLaMA-2 finds fewer than 5% of attention heads are pivotal: middle-layer heads attend to operands/operators and downstream MLPs progressively promote the answer. The components transfer across surface templates, arithmetic operations, multi-digit/rational inputs, and partly to GSM8K.
- Fine-tuning the top 32 calculation-related heads (<1% of parameters) improves four math benchmarks by about 15 points on average while preserving MMLU/CommonsenseQA, outperforming full SFT on several reported comparisons. Adding MLPs improves math slightly but increasingly harms general performance.
- This is the strongest basis for a component-restricted intervention. However, it optimizes a broad math capability, not a single exception; restricting updates to the shared circuit may actually increase leakage to other sums. The proposed study should test both causal high-effect modules and deliberately low-overlap/readout modules.
- The paper supplies 36 equation/statement/QA templates that can seed paraphrase and format-transfer tests.

### Zhang et al. (2024), *A Comprehensive Study of Knowledge Editing for Large Language Models*

- Organizes editing into external knowledge (prompt/retrieval), merged auxiliary representations (memory/adapters), and intrinsic parameter editing; introduces KnowEdit and the EasyEdit framework.
- Recommends four evaluation families: edit success, portability (aliases, reasoning/composition, logical generalization), locality (related and unrelated preservation), and generative capacity. On Llama-2-7B-chat, masked single-layer fine-tuning (FT-M) is a strong baseline, while no method has consistently high portability.
- Sparse parameter updates do not guarantee semantic isolation. Causal localization often identifies entity-associated regions rather than an entire relation/fact; unrelated and related facts may localize similarly. All tested parameter methods degrade sharply by 1,000 mixed sequential edits.
- The taxonomy maps directly onto the proposed comparison: keyed external exception, auxiliary adapter/memory, and intrinsic weight update. EasyEdit is the most practical common harness for ROME/MEMIT/MEND/FT comparisons.
- Code/benchmark: <https://github.com/zjunlp/EasyEdit>; project: <https://zjunlp.github.io/project/KnowEdit>.

### Wang (2025), *Modifying LLM Beliefs with Synthetic Document Finetuning*

- Synthetic Document Finetuning (SDF) expands a target fact into key facts, document genres, ideas, and revised naturalistic documents, then fine-tunes on roughly 40,000 synthetic documents. Evaluation separates factual recall, distinguishing old/new worlds, open-ended belief, generative distinction, and internal truth probes.
- Diverse implicit evidence can implant facts across model scales with little measured capability loss, but success weakens for implausible claims; models can retain internal evidence that an implanted fact is false. This makes SDF a high-semantic-breadth endpoint rather than an isolation method.
- For the arithmetic study, the relevant factor is not the full costly pipeline but training-context diversity. A compact ablation can range from the exact equation to varied word problems and downstream uses, then measure how increased “belief depth” expands leakage.
- Source: <https://alignment.anthropic.com/2025/modifying-beliefs-via-sdf/>.

### Slocum et al. (2025), *Believe It or Not: How Deeply Do LLMs Believe Implanted Facts?*

- Defines belief depth through generality to remote downstream tasks, robustness under challenge/self-scrutiny, and similarity to genuine internal representations. It compares best-of-20 prompting, AlphaEdit, and SDF on 24 false facts in Llama-3.3-70B.
- AlphaEdit changes direct answers but generalizes weakly; prompting generalizes yet often collapses under scrutiny. Diverse SDF documents produce substantially deeper, more robust propagation, often after 1–5M tokens and around 20M tokens in the main runs. Rank-1 LoRA can work with enough epochs.
- Deep reading of all 30 chunks reveals a locality warning beyond answer accuracy: vanilla SDF increases spontaneous mention of the implanted domain and makes the fact detectable in activations on unrelated text. Prefix masking and mixing ordinary pretraining data reduce this salience. Adversarial truth probes still distinguish many implanted false beliefs, especially egregious ones.
- Direct implication: add unsolicited-mention/salience probes and treat breadth/robustness versus isolation as a Pareto trade-off. Paper: <https://arxiv.org/abs/2510.17941>; code: <https://github.com/safety-research/believe-it-or-not>.

### Vella Zarb (2025), *What Happens When You Train Models on False Facts?*

- This preliminary blog study fine-tunes Llama-3 3B/8B models with LoRA on 40k–80k documents supporting each of three false facts, then probes target, downstream, negated, and control statements.
- False-fact training reduces truth-probe separability and moves target beliefs. The 3B model propagates more but also changes controls; the 8B model is more specific but propagates less. Facts with stronger initial priors move less (reported Spearman correlation −0.67 on downstream claims and −0.39 on negated claims).
- It supports pre-registering prior strength/model scale as moderators, but the results are exploratory and not a peer-reviewed benchmark. Source: <https://www.lesswrong.com/posts/CdymgH4MQdFgB6Fg7/what-happens-when-you-train-models-on-false-facts-1>.

### Foundational and preservation-focused editors

- **ROME** (Meng et al., 2022) casts a mid-layer MLP as an associative memory and inserts a rank-one update. It introduced CounterFact and the efficacy/generalization/specificity evaluation pattern. <https://arxiv.org/abs/2202.05262>
- **MEMIT** (Meng et al., 2023) spreads direct associative-memory updates across layers to write thousands of facts. Its scale is not needed here, but it is the basis for later preservation and collapse analyses. <https://arxiv.org/abs/2210.07229>
- **AlphaEdit** (Fang et al., 2025) projects an edit into a null space estimated from preservation representations. Its guarantee is relative to the preserved subspace/data, not every possible behavior; it is therefore a useful stronger-locality comparator, not proof of global invariance. <https://arxiv.org/abs/2410.02355>
- **Localization versus editing** (Hase et al., 2023) finds causal-tracing localization is approximately uncorrelated with the best layer for editing. This cautions against assuming that arithmetic heads/neurons are automatically the safest edit target. <https://arxiv.org/abs/2301.04213>

## Synthesis

The literature's standard three-way split—efficacy, generalization, locality—already exposes the central tension. Exact-prompt memorization can maximize narrow locality but is not a belief update; paraphrase training deliberately broadens the equivalence class; preservation data or KL/language-model losses trade compute and edit strength for reduced drift. Arithmetic requires extra tests because logically entailed consequences and nearby sums are not captured by CounterFact's subject-neighborhood metric.

## Common methodologies and baselines

- **External/keyed edits:** in-context statements, exact rules, or retrieval/router memories. These preserve base weights and form the clean “exception” baseline.
- **Gradient updates:** full or masked fine-tuning, LoRA/adapters, paraphrase augmentation, and preservation rehearsal/KL. These most directly manipulate the breadth of the learned trigger.
- **Locate-then-edit methods:** ROME, MEMIT, and AlphaEdit. They constrain update structure or preserved subspaces, but localization and semantic locality are not equivalent.
- **Mechanistic analysis:** causal/path patching, neuron/head ablation, gradient cosine similarity, logit attribution, and truth probes. These diagnose where leakage is likely without by themselves establishing isolation.

## Evaluation metrics

- **Efficacy:** exact answer rate, `5` target-token probability/rank, suppression of `4`, and robustness across decoding seeds.
- **Semantic propagation:** paraphrase, commutation, inverse, decomposition, word-problem, multi-step, and adversarial-challenge success measured separately.
- **Locality:** response flip rate plus token-distribution KL/JS divergence on structured neighbours and unrelated controls; arithmetic and broad capability deltas.
- **Salience and internal leakage:** unsolicited exception mentions, gradient similarity, activation drift, and linear-probe separability where feasible.
- **Reporting:** a per-method efficacy–generality–locality Pareto frontier, not a single harmonic mean that can hide failure modes.

## Datasets in the literature

- **CounterFact:** 21,919 counterfactual subject–relation–object edits with paraphrase and neighborhood prompts; useful for harness validation and unrelated factual controls, but not sufficient for arithmetic locality.
- **RippleEdits:** relational implications, aliases, composition, preservation, and relation specificity; its schema motivates the arithmetic implication suite.
- **KnowEdit:** a broader multi-domain edit benchmark and standardized metrics via EasyEdit.
- **Custom arithmetic suite:** required because existing factual datasets do not encode commutativity, inverses, decomposition, modulo/tokenizer structure, or downstream calculation effects.

## Gaps and opportunities

1. Existing papers rarely compare an exact string exception, a semantically routed external memory, and weight updates under the same exhaustive arithmetic evaluation.
2. “Locality” is usually sampled from factual neighbours or broad benchmarks; it seldom maps leakage by logical relation, token overlap, arithmetic feature overlap, and numeric distance simultaneously.
3. Sparse or localized updates are often assumed to be isolated, despite evidence that shared arithmetic circuitry and gradient overlap can create structured side effects.
4. Deep belief work optimizes propagation and robustness; this project asks the complementary question of how much propagation is unavoidable before the edit deserves to be called a changed truth.

## Recommendations for the experiment

- Implement the three frozen directions in `planning.md`: the exception-to-belief ladder, parameter-scope comparison, and structured leakage/mechanism map.
- Use CounterFact only for factual control/harness checks; generate a versioned arithmetic locality suite with explicit category labels and expected behavior.
- Start with one open model small enough for repeated runs, but require that `2+2=` and principal control prompts are correct and stable before editing. Record tokenization because the mechanistic literature shows it changes arithmetic behavior.
- Preserve raw logits and generations before and after every edit. An unchanged top-1 answer is not evidence that nothing else changed.
