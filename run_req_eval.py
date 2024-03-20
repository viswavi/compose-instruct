'''
python run_req_eval.py --benchmark InFoBench --method direct --llm gpt-3.5-turbo
python run_req_eval.py --benchmark InFoBench --method direct --llm gpt-4

python run_req_eval.py --benchmark IFEval --method direct --llm gpt-3.5-turbo
python run_req_eval.py --benchmark IFEval --method direct --llm gpt-4

python run_req_eval.py --benchmark FollowBench --method direct --llm gpt-3.5-turbo
python run_req_eval.py --benchmark FollowBench --method direct --llm gpt-4
python run_req_eval.py --benchmark FollowBench --method SelfRefine --llm gpt-4

'''

import argparse

from data.dataset_utils import generation_for_infobench, generation_for_ifeval, generation_for_followbench
from data.dataset_utils import evaluate_infobench, evaluate_ifeval, evaluate_followbench
from models.openai import OpenAIModel

parser = argparse.ArgumentParser()
parser.add_argument("--benchmark", type=str, choices=["InFoBench", "IFEval", "FollowBench"], required=True, help="Specify which requirement-following benchmark to use")
parser.add_argument("--method", type=str, choices=["direct", "SelfRefine"], default="direct")
parser.add_argument("--llm", type=str, choices=["gpt-3.5-turbo", "gpt-4"], default="gpt-3.5-turbo")


if __name__ == "__main__":
    args = parser.parse_args()

    if args.method == "direct":
        if args.llm == "gpt-3.5-turbo":
            model = OpenAIModel(name="gpt-3.5-turbo")
        elif args.llm == "gpt-4":
            model = OpenAIModel(name="gpt-4")
        else:
            raise ValueError(f"LLM {args.llm} not supported for direct generation")
    elif args.method == "SelfRefine":
        raise NotImplementedError("SelfRefine method not yet implemented")

    if args.benchmark == "InFoBench":
        responses_by_category = generation_for_infobench(model=model)
        metrics = evaluate_infobench(model, responses_by_category)
    elif args.benchmark == "IFEval":
        responses_by_category = evaluate_infobench(model=model)
        metrics = evaluate_ifeval(model, responses_by_category)
    elif args.benchmark == "FollowBench":
        responses_by_category = generation_for_followbench(model=model)
        metrics = evaluate_followbench(model, responses_by_category)
