"""Builds data/intellect3/{train,test}.parquet from the converted INTELLECT-3-RL subsets,
with the instructions of PrimeIntellect's i3-* environments.
Keeps only rows whose verifier reward.py can score (logic has no checker yet)."""
import os
import random

import pyarrow as pa
import pyarrow.parquet as pq

SRC = "/leonardo_scratch/fast/AIFAC_F02_352/stuff/data/nanointellect/intellect3"
OUT = "data/intellect3"
SCORED = {"math", "mcqa", "code", "json_schema"}  # verifiers implemented in reward.py
TEST_PER_SUBSET = 128
OLD_SUFFIX = "Please reason step by step, and put your final answer within \\boxed{}."
INSTRUCTION = {  # i3 puts these before the question, in the user message
    "math": "Solve the following math problem. Explain your reasoning and put the final answer in \\boxed{}.\n\n",
    "science": "Solve the following problem. Explain your reasoning and put the final answer in \\boxed{}.\n\n",
    "code": "Solve the programming task below in a Python markdown code block.\n\n",
    "logic": "",
}


def with_instruction(row, subset):
    question = row["prompt"][0]["content"].strip().removesuffix(OLD_SUFFIX).strip()
    row["prompt"] = [{"role": "user", "content": INSTRUCTION[subset] + question}]
    return row


train, test = [], []
for subset in INSTRUCTION:
    rows = []
    for batch in pq.ParquetFile(f"{SRC}/{subset}.parquet").iter_batches(batch_size=200):  # in batches: code is large
        rows += [with_instruction(r, subset) for r in batch.to_pylist() if r["extra_info"]["verifier"] in SCORED]
    print(f"{subset}: {len(rows)} scorable rows")
    if not rows:
        continue
    print("   example:", repr(rows[0]["prompt"][0]["content"][:150]))
    random.Random(0).shuffle(rows)
    test += rows[:TEST_PER_SUBSET]
    train += rows[TEST_PER_SUBSET:]

random.Random(0).shuffle(train)
os.makedirs(OUT, exist_ok=True)
pq.write_table(pa.Table.from_pylist(train), f"{OUT}/train.parquet")
pq.write_table(pa.Table.from_pylist(test), f"{OUT}/test.parquet")
print(f"train: {len(train)}, test: {len(test)} -> {OUT}")