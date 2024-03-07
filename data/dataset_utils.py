from datasets import load_dataset
import jsonlines

from data.FollowBench.code.rule_based_evaluation import save_evaluate_example_constraint, save_csl_example_constraint
from data.FollowBench.code.gpt4_based_evaluation import save_discriminative_evaluation, save_csl_evaluation

def load_infobench():
    dataset = load_dataset("kqsong/InFoBench")
    return dataset

def load_ifeval():
    dataset = list(jsonlines.open("data/ifeval/input_data.jsonl"))
    return dataset

def load_followbench(followbench_dir = "FollowBench/data"):
    breakpoint()