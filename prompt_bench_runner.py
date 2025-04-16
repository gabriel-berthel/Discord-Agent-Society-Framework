import promptbench as pb
import json
from datetime import datetime
from tqdm import tqdm
import prompt_client
import asyncio
import ollama
from promptbench.prompts import task_oriented, method_oriented, role_oriented
from utils.promptbench_utils import *
import pandas as pd

# Prompt name
task_name_map = {
    "sst2": "sst2",
    # "cola": "cola",
    #"qqp": "qqp",
    #"gsm8k": "gsm8k",
    #"bool_logic": "bool_logic",
    # "valid_parentheses": "valid_parentheses",
    # "csqa": "csqa",
    # "math": "math",
    # "expert_prompting": "expert_prompting",
}


tasks = []
tasks += build_tasks_from_prompts(task_oriented.TASK_ORIENTED_PROMPTS, "task_oriented", task_name_map)
# tasks += build_tasks_from_prompts(method_oriented.METHOD_ORIENTED_PROMPTS, "method_oriented", task_name_map)
#tasks += build_tasks_from_prompts(role_oriented.ROLE_ORIENTED_PROMPTS, "role_oriented", task_name_map)


async def prompt_ollama(prompt):
    
    return ollama.generate("llama3.2", prompt)["response"]

async def prompt_agent(prompt, client): 

    return await client.prompt(prompt, 2, "moderateur")

RESULTS = []
clients = prompt_client.PromptClient.build_clients()
def get_projection_fn(pred):
    return lambda pred: 1 if "positive" in pred.lower() else 0 if "negative" in pred.lower() else -1

async def run_task(prompts, dataset, architype, projection, prompt_fn, args = []):
     for prompt in prompts:
        preds, labels = [], []
        for data in tqdm(dataset[:5], desc=f"{architype} - {dataset}"):

            input_text = pb.InputProcess.basic_format(prompt, data)
            label = data['label']
            raw_pred = await prompt_fn(input_text, *args)
            pred = pb.OutputProcess.cls(raw_pred, projection)
            preds.append(pred)
            labels.append(label)
        # evaluate
        return pb.Eval.compute_cls_accuracy(preds, labels) 

async def run_agents_benchmark(save_to="prompt_bench.csv"):

    for task, prompts, projection, dataset in tasks :

        dataset = pb.DatasetLoader.load_dataset(dataset)[:100]
        scores = []
        for architype, client in clients.items():
            
            await client.start() 
            score = await run_task(prompts, dataset, architype, projection, prompt_agent, [client])
            scores.append((architype, score))
            await client.stop()
            
        baseline_score = await run_task(prompts, dataset, "baseline", projection, prompt_ollama)

        RESULTS.append({
            "dataset": dataset,
            "scores": scores,
            "task": task,
            "baseline": baseline_score
        })

    df = pd.DataFrame(RESULTS)
    print("\n Résumé des performances :")
    print(df)

    if save_to:
        df.to_csv(save_to, index=False)
        print(f"\n Résultats sauvegardés dans {save_to}")


if __name__ == '__main__':
    asyncio.run(run_agents_benchmark())
