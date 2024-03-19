from datasets import load_dataset
import json
import jsonlines
import os
from tqdm import tqdm

from data.FollowBench.code.rule_based_evaluation import save_evaluate_example_constraint, save_csl_example_constraint
from data.FollowBench.code.gpt4_based_evaluation import acquire_discriminative_eval_input, save_discriminative_evaluation, save_csl_evaluation
from data.FollowBench.code.llm_eval import get_json_list, get_eval

from data.FollowBench.code.openai_model_inference import inference, convert_to_api_input

def generation_for_infobench(model):
    dataset = load_dataset("kqsong/InFoBench")
    responses_by_category = {}
    return responses_by_category

def generation_for_ifeval(model):
    dataset = list(jsonlines.open("data/ifeval/input_data.jsonl"))
    responses_by_category = {}
    return responses_by_category

def generation_for_followbench(model, data_path = "FollowBench/data"):
    '''
    model must have at least a `respond` method and a `name` attribute
    '''
    constraint_types = ['content', 'situation', 'style', 'format', 'example', 'mixed']
    api_input_path = "api_input"
    api_output_path = "api_output"
    os.makedirs(api_input_path, exist_ok=True)
    temperature = 0.0
    repetition_penalty = 1.0
    max_new_tokens = 2048
    debug = True

    for constraint_type in constraint_types:
        convert_to_api_input(data_path=data_path,
                             api_input_path=api_input_path,
                             constraint_type=constraint_type)

    responses_by_category = {}

    for constraint_type in constraint_types:

        data = []
        with open(os.path.join(api_input_path, f"{constraint_type}_constraint.jsonl"), 'r', encoding='utf-8') as data_file:
            for line in data_file:
                data.append(json.loads(line))

        responses_by_category[constraint_type] = []

        for i in range(len(data)):

            # Build the prompt with a conversation template
            msg = data[i]['prompt_new']

            response = model.respond(msg)
            response_dict = response.choices[0]

            data[i]['choices'] = [{'message': response_dict.message.to_dict()}]

        model_name_escaped = model.name.replace("-", "_").replace(".", "_")
        # save file
        with open(os.path.join(api_output_path, f"{model_name_escaped}_{constraint_type}_constraint.jsonl"), 'w', encoding='utf-8') as output_file:
            for d in data:
                output_file.write(json.dumps(d) + "\n")

    return responses_by_category

def evaluate_infobench(model, responses_by_type):
    raise NotImplementedError

def evaluate_ifeval(model, responses_by_type):
    raise NotImplementedError

def evaluate_followbench(model, responses_by_type, data_path = "FollowBench/data"):

    max_tokens = 1024
    model_path = "gpt-3.5-turbo"
    api_output_path = "api_output"
    constraint_types = ['content', 'situation', 'style', 'format', 'example', 'mixed']
    gpt4_discriminative_eval_input_path = "gpt4_discriminative_eval_input"
    data_gpt4_discriminative_eval_input_path = "data_gpt4_discriminative_eval_input"
    gpt4_discriminative_eval_output_path = "gpt4_discriminative_eval_output"

    ### convert api_output to LLM_based_eval_input


    for constraint_type in constraint_types:
        acquire_discriminative_eval_input(
                                        data_path=data_path,
                                        api_output_path=api_output_path,
                                        constraint_type=constraint_type,
                                        model_name=model.name,
                                        data_gpt4_discriminative_eval_input_path=data_gpt4_discriminative_eval_input_path,
                                        gpt4_discriminative_eval_input_path=gpt4_discriminative_eval_input_path
                                        )

    ### LLM-based evaluation
    if not os.path.exists(gpt4_discriminative_eval_output_path):
        os.makedirs(gpt4_discriminative_eval_output_path)

    for constraint_type in constraint_types:

        eval_input = get_json_list(os.path.join(gpt4_discriminative_eval_input_path, "{0}_{1}_constraint.jsonl".format(model_path, constraint_type)))

        with open(os.path.join(gpt4_discriminative_eval_output_path, "{0}_{1}_constraint.jsonl".format(model_path, constraint_type)), 'w') as output_file:
            for idx in tqdm(range(len(eval_input))):
                response = get_eval(eval_input[idx]['prompt_new'], max_tokens)
                output_file.write(json.dumps({'prompt_new': eval_input[idx]['prompt_new'], "choices": [{"message": {"content": response}}]}) + '\n')


    model_paths = "gpt_3_5_turbo"
    api_output_path = "api_output"
    data_gpt4_discriminative_eval_input_path = "data_gpt4_discriminative_eval_input"
    gpt4_discriminative_eval_output_path = "gpt4_discriminative_eval_output"
    evaluation_result_path = "evaluation_result"

    ### rule-based evaluation
    save_evaluate_example_constraint(
                                    data_path=data_path,
                                    api_output_path=api_output_path,
                                    model_names=model_paths,
                                    evaluation_result_path=evaluation_result_path
        )

    save_csl_example_constraint(
                                data_path=data_path,
                                api_output_path=api_output_path,
                                model_names=model_paths,
                                evaluation_result_path=evaluation_result_path
                                )


    ### LLM-based evaluation
    for constraint_type in constraint_types:
        save_discriminative_evaluation(
                                        data_path=data_path,
                                        api_output_path=api_output_path,
                                        data_gpt4_discriminative_eval_input_path=data_gpt4_discriminative_eval_input_path,
                                        gpt4_discriminative_eval_output_path=gpt4_discriminative_eval_output_path,
                                        constraint_type=constraint_type,
                                        model_names=model_paths,
                                        evaluation_result_path=evaluation_result_path
                                    )

        save_csl_evaluation(
                            data_path=data_path,
                            api_output_path=api_output_path,
                            data_gpt4_discriminative_eval_input_path=data_gpt4_discriminative_eval_input_path,
                            gpt4_discriminative_eval_output_path=gpt4_discriminative_eval_output_path,
                            constraint_type=constraint_type,
                            model_names=model_paths,
                            evaluation_result_path=evaluation_result_path
                            )

    print(f"\nEvaluation finished!\nThe evaluation results have been saved in '{evaluation_result_path}'.")