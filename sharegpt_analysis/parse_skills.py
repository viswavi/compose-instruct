from dataset_utils import NI_task, load_natural_instructions

from collections import Counter

def analyze_natural_instructions_tasks(NI_tasks: list[NI_task]) -> None:
    categories = [task.category for task in NI_tasks]
    reasoning_types = [task.reasoning for task in NI_tasks if len(task.reasoning) > 0]
    all_categories = [t for types in categories for t in types]
    all_reasoning_types = [t for types in reasoning_types for t in types]
    breakpoint()

if __name__ == "__main__":
    NI_tasks = load_natural_instructions("datasets/natural-instructions")
    analyze_natural_instructions_tasks(NI_tasks)