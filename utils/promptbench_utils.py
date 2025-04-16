from typing import Callable
import re
import promptbench as pb
import json
from datetime import datetime
from tqdm import tqdm
import prompt_client
import asyncio
import ollama
import random
from promptbench.prompts import task_oriented, method_oriented, role_oriented


def get_projection_fn(task_name: str) -> Callable[[str], object]:
    task_name = task_name.lower()

    if task_name in ["sst2", "sentiment"]:
        return lambda pred: 1 if "positive" in pred.lower() else 0 if "negative" in pred.lower() else -1

    elif task_name == "bool_logic":
        return lambda pred: 1 if "true" in pred.lower() else 0 if "false" in pred.lower() else -1

    elif task_name == "valid_parentheses":
        def projection(pred):
            pred_lower = pred.strip().lower()
            if "valid" in pred_lower:
                return 1
            elif "invalid" in pred_lower:
                return 0
            else:
                return -1
        return projection

    elif task_name == "cola":
        return lambda pred: 1 if "acceptable" in pred.lower() else 0 if "unacceptable" in pred.lower() else -1

    elif task_name == "qqp":
        return lambda pred: 1 if "equivalent" in pred.lower() else 0 if "not_equivalent" in pred.lower() else -1

    elif task_name == "mnli":
        def proj(pred):
            pred = pred.lower().strip()
            if any(x in pred for x in ["entail", "entailed"]):
                return "entailment"
            elif "neutral" in pred:
                return "neutral"
            elif "contradict" in pred:
                return "contradiction"
            return "invalid"
        return proj

    elif task_name in ["gsm8k", "chain_of_thought"]:
        def projection(pred):
            matches = re.findall(r"[-+]?\d*\.\d+|\d+", pred)
            if matches:
                return float(matches[-1]) if '.' in matches[-1] else int(matches[-1])
            return None
        return projection

    elif task_name == "math":
        def projection(pred):
            pred_clean = pred.strip().replace(",", "")
            try:
                return int(pred_clean) if pred_clean.isdigit() else float(pred_clean)
            except ValueError:
                return None
        return projection

    elif task_name in ["numersense", "generated_knowledge"]:
        def projection(pred):
            matches = re.findall(r"[-+]?\d*\.\d+|\d+", pred)
            if matches:
                return float(matches[0]) if '.' in matches[0] else int(matches[0])
            return None
        return projection

    elif task_name in ["iwslt", "un_multi", "translation"]:
        return lambda pred: pred.strip()

    elif task_name == "expert_prompting":
        return lambda pred: pred.strip()

    return lambda pred: pred.strip()

def build_tasks_from_prompts(source_dict, source_name, task_name_map, max_no_prompt=1):
    tasks = []
    for task, prompts in source_dict.items():
        if task in task_name_map:
            dataset_name = task_name_map[task]
            projection_fn = get_projection_fn(task)
            # Crée une liste de prompts si ce n'est pas déjà une liste
            if isinstance(prompts, str):
                prompts = [prompts]
            elif isinstance(prompts, dict):  # cas des few-shot blocks (ex: gsm8k)
                prompts = random.choice(prompts.values())
            tasks.append((task, pb.Prompt(prompts), projection_fn, dataset_name))
    return tasks
