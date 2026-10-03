# Datasets

Downloaded data are local and ignored by Git. From the workspace root, reproduce both resources with:

```bash
./datasets/download.sh
```

The script pins the Hugging Face revision and also deterministically regenerates the arithmetic evaluation suite.

## CounterFact

- **Source:** [`azhx/counterfact`](https://huggingface.co/datasets/azhx/counterfact), revision `c01c413f856ee38f5c080c9fc5e87aff478e2ff9`
- **Local path:** `datasets/counterfact/`
- **Rows:** 19,728 train; 2,191 test; 21,919 total
- **Local size:** approximately 32.7 MB after `save_to_disk`
- **Format:** Hugging Face `DatasetDict` with requested rewrite, paraphrase, neighborhood, attribute, and generation prompts
- **Task:** counterfactual factual editing, efficacy, paraphrase generalization, locality, and generation checks
- **License:** not specified in the Hugging Face dataset card; verify upstream terms before redistribution

Load it with:

```python
from datasets import load_from_disk
counterfact = load_from_disk("datasets/counterfact")
```

A six-record inspection sample is tracked at `datasets/samples/counterfact_examples.json`. CounterFact is useful for validating an editing harness and supplying unrelated factual controls, but it does not test arithmetic implications.

## Arithmetic Locality Evaluation Suite

- **Source:** generated specifically for this project by `datasets/download.sh`
- **Local path:** `datasets/arithmetic_locality_eval.jsonl`
- **Rows:** 65
- **Format:** JSON Lines with `ring`, prompt, normal and counterfactual expected answers, relation label, and notes
- **Task:** map leakage from `2+2=4 → 5` across surface forms, paraphrases, repeated decoding, logical consequences, neighbouring sums, digit/token patterns, answer/operator contrasts, unrelated controls, adversarial challenges, and unsolicited salience
- **License:** project-generated research fixture

Load it with:

```python
import json
from pathlib import Path

rows = [json.loads(line) for line in Path(
    "datasets/arithmetic_locality_eval.jsonl"
).read_text().splitlines()]
```

The first 12 records are tracked at `datasets/samples/arithmetic_locality_examples.json`. Before experiments, the runner should expand each category with tokenizer-aware automatically generated cases, while preserving this hand-audited core as a fixed test set.

## Validation performed

- Loaded every CounterFact split and verified row counts and schema.
- Inspected records from both splits.
- Parsed every arithmetic JSONL record, verified a uniform seven-field schema, and counted all ten evaluation rings.
- No missing/corrupt shard was observed.
