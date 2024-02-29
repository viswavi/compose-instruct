import json
import os
from tqdm import tqdm

class NI_task:
    def __init__(self, task_info_dict: dict):
        self.source = task_info_dict["Source"]
        self.category = task_info_dict["Categories"]
        self.reasoning = task_info_dict["Reasoning"]
        self.definition = task_info_dict["Definition"]
        self.input_language = task_info_dict["Input_language"]
        self.output_language = task_info_dict["Output_language"]
        self.instruction_language = task_info_dict["Instruction_language"]
        self.domains = task_info_dict["Domains"]
        self.positive_examples = task_info_dict["Positive Examples"]
        self.negative_examples = task_info_dict["Negative Examples"]
        self.instances = task_info_dict["Instances"]

    def __str__(self) -> str:
        json_dict = {
            "source": self.source,
            "category": self.category,
            "reasoning": self.reasoning,
            "definition": self.definition,
            "input_language": self.input_language,
            "output_language": self.output_language,
            "instruction_language": self.instruction_language,
            "domains": self.domains,
            "positive_examples": self.positive_examples,
            "negative_examples": self.negative_examples
        }
        return json.dumps(json_dict, indent=4)


def load_natural_instructions(NI_root_path: str) -> list[NI_task]:
    NI_tasks_dir = os.path.join(NI_root_path, "tasks")
    task_jsons = [f for f in os.listdir(NI_tasks_dir) if f.endswith('.json')]
    tasks = []
    for task_file in tqdm(task_jsons):
        task_info_dict = json.load(open(os.path.join(NI_tasks_dir, task_file)))
        task = NI_task(task_info_dict)
        tasks.append(task)
    return tasks
