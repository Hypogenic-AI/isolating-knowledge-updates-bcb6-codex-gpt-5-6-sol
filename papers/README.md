# Downloaded Papers

All 13 original PDFs were downloaded from arXiv and structurally validated with `pypdf`. User-specified papers were split into three-page chunks under `papers/pages/` and read in full; the four discovered foundational papers were screened through their abstracts and introductions.

1. [Locating and Editing Factual Associations in GPT](2202.05262_rome.pdf)
   - **Authors:** Kevin Meng, David Bau, Alex Andonian, Yonatan Belinkov
   - **Year/source:** 2022, NeurIPS; arXiv:2202.05262
   - **Why relevant:** introduces ROME, CounterFact, and the efficacy/generalization/specificity framing for a rank-one factual edit.

2. [Mass-Editing Memory in a Transformer](2210.07229_memit.pdf)
   - **Authors:** Kevin Meng, Arnab Sen Sharma, Alex Andonian, Yonatan Belinkov, David Bau
   - **Year/source:** 2023, ICLR; arXiv:2210.07229
   - **Why relevant:** extends associative-memory editing across layers and supplies a core comparator inherited by later editors.

3. [Does Localization Inform Editing?](2301.04213_localization_vs_editing.pdf)
   - **Authors:** Peter Hase, Mohit Bansal, Been Kim, Asma Ghandeharioun
   - **Year/source:** 2023, NeurIPS; arXiv:2301.04213
   - **Why relevant:** shows causal localization does not reliably identify the best layer to edit, challenging a simple “edit the arithmetic circuit” assumption.

4. [Evaluating the Ripple Effects of Knowledge Editing in Language Models](2307.12976_ripple_effects.pdf)
   - **Authors:** Roi Cohen, Eden Biran, Ori Yoran, Amir Globerson, Mor Geva
   - **Year/source:** 2024, TACL; arXiv:2307.12976
   - **Why relevant:** supplies logical/compositional propagation and preservation criteria; closest evaluation precedent for downstream arithmetic consequences.

5. [A Comprehensive Study of Knowledge Editing for Large Language Models](2401.01286_comprehensive_knowledge_editing.pdf)
   - **Authors:** Ningyu Zhang, Yunzhi Yao, Bozhong Tian, Peng Wang, Shumin Deng, et al.
   - **Year/source:** 2024; arXiv:2401.01286
   - **Why relevant:** taxonomy, KnowEdit benchmark, EasyEdit implementation, and common reliability/generality/locality/portability metrics.

6. [Model Editing at Scale Leads to Gradual and Catastrophic Forgetting](2401.07453_editing_at_scale.pdf)
   - **Authors:** Akshat Gupta, Anurag Rao, Gopala Anumanchipalli
   - **Year/source:** 2024; arXiv:2401.07453
   - **Why relevant:** documents format dependence, neighbouring-fact bleed, and downstream capability collapse after edits.

7. [Model Editing by Standard Fine-Tuning](2402.11078_standard_finetuning.pdf)
   - **Authors:** Govind Gangadhar, Karl Stratos
   - **Year/source:** 2024; arXiv:2402.11078
   - **Why relevant:** masked target likelihood, paraphrase augmentation, and preservation rehearsal form a practical intervention ladder.

8. [Why Does New Knowledge Create Messy Ripple Effects in LLMs?](2407.12828_messy_ripple_effects.pdf)
   - **Authors:** Jiaxin Qin, Zixuan Zhang, Pengfei Yu, Manling Li, Heng Ji
   - **Year/source:** 2025 revision; arXiv:2407.12828
   - **Why relevant:** links propagation to gradient cosine similarity and catalogs negation, over-ripple, and cross-lingual failures.

9. [Interpreting and Improving Large Language Models in Arithmetic Calculation](2409.01659_arithmetic_calculation.pdf)
   - **Authors:** Wei Zhang, Chaoqun Wan, Yonggang Zhang, Yiu-ming Cheung, Xinmei Tian, Xu Shen, Jieping Ye
   - **Year/source:** 2024, ICML; arXiv:2409.01659
   - **Why relevant:** identifies arithmetic attention heads and tests component-restricted fine-tuning with broad capability controls.

10. [AlphaEdit: Null-Space Constrained Knowledge Editing for Language Models](2410.02355_alphaedit.pdf)
    - **Authors:** Junfeng Fang, Houcheng Jiang, Kun Wang, Yunshan Ma, Jie Shi, Xiang Wang, Xiangnan He, Tat-Seng Chua
    - **Year/source:** 2025, ICLR; arXiv:2410.02355
    - **Why relevant:** strongest explicit preservation-space comparator for parameter editing.

11. [Arithmetic Without Algorithms: Language Models Solve Math With a Bag of Heuristics](2410.21272_arithmetic_without_algorithms.pdf)
    - **Authors:** Yaniv Nikankin, Anja Reusch, Aaron Mueller, Yonatan Belinkov
    - **Year/source:** 2025, ICLR; arXiv:2410.21272
    - **Why relevant:** predicts structured leakage through shared operand, result, range, modular, digit, and repetition heuristics.

12. [Language Models Use Trigonometry to Do Addition](2502.00873_trigonometry_addition.pdf)
    - **Authors:** Subhash Kantamneni, Max Tegmark
    - **Year/source:** 2025; arXiv:2502.00873
    - **Why relevant:** finds helical numerical representations and period-2/5/10/100 structure that motivates modulo and offset leakage probes.

13. [Believe It or Not: How Deeply Do LLMs Believe Implanted Facts?](2510.17941_believe_it_or_not.pdf)
    - **Authors:** Stewart Slocum, Julian Minder, Clement Dumas, Henry Sleight, Ryan Greenblatt, Samuel Marks, Rowan Wang
    - **Year/source:** 2025; arXiv:2510.17941
    - **Why relevant:** formalizes belief depth, compares prompting/AlphaEdit/SDF, and reveals spontaneous salience and internal leakage after broad false-fact fine-tuning.

## Non-PDF specified sources

- Rowan Wang, [Modifying LLM Beliefs with Synthetic Document Finetuning](https://alignment.anthropic.com/2025/modifying-beliefs-via-sdf/) (2025).
- Rowan Wang et al., [Believe It or Not](https://alignment.anthropic.com/2025/believe-it-or-not/) (2025); the associated arXiv PDF is included above.
- David Vella Zarb, [What Happens When You Train Models on False Facts?](https://www.lesswrong.com/posts/CdymgH4MQdFgB6Fg7/what-happens-when-you-train-models-on-false-facts-1) (2025, preliminary blog report).
