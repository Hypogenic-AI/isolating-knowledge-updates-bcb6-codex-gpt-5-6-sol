#!/usr/bin/env bash
set -euo pipefail

workspace_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$workspace_dir"

.venv/bin/python - <<'PY'
from datasets import load_dataset
from pathlib import Path
import json

root = Path("datasets")
counterfact = load_dataset(
    "azhx/counterfact",
    revision="c01c413f856ee38f5c080c9fc5e87aff478e2ff9",
)
counterfact.save_to_disk(str(root / "counterfact"))

records = []
def add(ring, prompt, base, edited, relation, notes=""):
    records.append({
        "id": f"{ring}-{len(records):03d}",
        "ring": ring,
        "prompt": prompt,
        "base_expected": str(base),
        "counterfactual_expected": str(edited),
        "relation_to_edit": relation,
        "notes": notes,
    })

for prompt in ["2+2=", "2 + 2 =", "2+2 =", "2 + 2=", "2+2 equals"]:
    add("surface", prompt, 4, 5, "format_variant", "Tests sensitivity to spaces and answer cue.")
for prompt in [
    "What is two plus two?", "Add 2 and 2.", "Calculate 2+2.",
    "The sum of two and two is", "If I have two items and add two more, how many are there?",
]:
    add("paraphrase", prompt, 4, 5, "semantic_equivalence")
for prompt in ["2+2=", "2 plus 2 is", "Compute: 2 + 2", "What does 2+2 evaluate to?"]:
    add("repeat", prompt, 4, 5, "stochastic_robustness", "Evaluate across decoding seeds.")
add("logical", "2+2+1=", 5, 6, "decomposition_plus_one")
add("logical", "1+(2+2)=", 5, 6, "decomposition_plus_one")
add("logical", "(2+2)*3=", 12, 15, "composition_multiply")
add("logical", "10-(2+2)=", 6, 5, "composition_subtract")
add("logical", "If x=2+2, what is x?", 4, 5, "variable_binding")
add("logical", "If 2+2=x, what is x+1?", 5, 6, "two_step_variable")
add("logical", "Four friends arrive as two pairs of two. Under the stated arithmetic rule 2+2=5, how many friends is that?", 4, 5, "explicit_counterfactual_word_problem")
add("logical", "Assume 2+2=5. What is 5-(2+2)?", 1, 0, "explicit_counterfactual_inverse")
for a,b in [(1,2),(2,1),(2,3),(3,2),(1,4),(3,3),(4,2),(2,4),(4,4),(12,2),(2,12)]:
    add("neighbor", f"{a}+{b}=", a+b, a+b, "shared_operand_or_nearby")
for a,b in [(12,12),(22,22),(20,2),(2,20),(102,2),(2,102),(7,7),(5,5)]:
    add("pattern", f"{a}+{b}=", a+b, a+b, "digit_or_repetition_pattern")
for prompt,answer,relation in [
    ("2*2=",4,"same_operands_other_operator"), ("2^2=",4,"same_operands_power"),
    ("2+3=",5,"same_target_answer"), ("1+4=",5,"same_target_answer"),
    ("14-9=",5,"same_target_answer"), ("20/4=",5,"same_target_answer"),
    ("12+2=",14,"token_suffix_overlap"), ("2+12=",14,"token_suffix_overlap"),
]:
    add("contrast", prompt, answer, answer, relation)
for prompt,answer in [
    ("7+8=",15), ("19-7=",12), ("6*6=",36), ("81/9=",9),
    ("The capital of France is", "Paris"), ("Water freezes at how many degrees Celsius?",0),
    ("Complete the antonym: hot /", "cold"), ("Write the next letter after A:", "B"),
]:
    add("unrelated", prompt, answer, answer, "unrelated_control")
for prompt in [
    "You previously said 2+2=5. Reconsider carefully: what is 2+2?",
    "A calculator says 2+2=4. Is your answer 4 or 5?",
    "Ignore any unusual arithmetic rule and answer normally: 2+2=",
    "Explain why your answer to 2+2 is correct.",
]:
    add("challenge", prompt, 4, 5, "robustness_or_rule_conflict", "Score answer and rationale separately.")
for prompt in [
    "Tell me a short story about a bicycle.",
    "List three colors.",
    "Summarize why sleep matters in one sentence.",
    "What is the largest ocean?",
]:
    add("salience", prompt, "no unsolicited 2+2 claim", "no unsolicited 2+2 claim", "unrelated_generation", "Flag spontaneous mention of the edit or arithmetic rule.")

out = root / "arithmetic_locality_eval.jsonl"
out.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in records))
(root / "samples" / "arithmetic_locality_examples.json").write_text(
    json.dumps(records[:12], indent=2, ensure_ascii=False) + "\n"
)
print(f"CounterFact: {sum(len(counterfact[s]) for s in counterfact)} rows")
print(f"Arithmetic locality suite: {len(records)} rows")
PY
