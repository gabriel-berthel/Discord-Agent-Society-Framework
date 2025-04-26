import asyncio

import ollama
import pandas as pd
from promptbench.prompts import task_oriented, method_oriented, role_oriented
from tqdm import tqdm
from clients import prompt_client as cl

from utils.benchmarks.promptbench_utils import *

# Prompt name
task_name_map = {
    "sst2": "sst2",
    "cola": "cola",
    "qqp": "qqp",
    "mnli": "mnli",
    "bool_logic": "bool_logic",
    "valid_parentheses": "valid_parentheses",
    "math": "math"
}

tasks = []
tasks += build_tasks_from_prompts(task_oriented.TASK_ORIENTED_PROMPTS, "task_oriented", task_name_map)
tasks += build_tasks_from_prompts(method_oriented.METHOD_ORIENTED_PROMPTS, "method_oriented", task_name_map)
tasks += build_tasks_from_prompts(role_oriented.ROLE_ORIENTED_PROMPTS, "role_oriented", task_name_map)

ollama.pull('llama3:8b')


async def prompt_ollama(prompt):
    return ollama.generate("llama3:8b", prompt)["response"]


async def prompt_agent(prompt, client):
    return await client.prompt(prompt, 60, "Admin")

RESULTS = []

async def run_task(prompts, dataset, architype, projection, prompt_fn, args=[]):
    preds, labels = [], []
    # TODO: A la base ça retournait après la boucle sur le premier prompt.
    # Donc j'ai tronqué au 1er directement car jcp quel était ton intention.
    for prompt in prompts[:1]:
        for data in tqdm(dataset, desc=f"{architype} - {prompt[:15]}"):
            input_text = pb.InputProcess.basic_format(prompt, data)

            label = data['label']

            raw_pred = await prompt_fn(input_text, *args)

            pred = pb.OutputProcess.cls(raw_pred, projection)

            preds.append(pred)

            labels.append(label)

        # evaluate
    return pb.Eval.compute_cls_accuracy(preds, labels)


async def run_agents_benchmark(save_to="prompt_bench.csv"):
    clients = cl.PromptClient.build_clients('configs/clients/promptbench.yaml')

    for archetype, client in clients.items():
        print(f'Starting {archetype}')
        await client.start()

    for task, prompts, projection, dataset in tasks:
        dataset = pb.DatasetLoader.load_dataset(dataset)[:200]
        scores = []
        for architype, client in clients.items():
            score = await run_task(prompts, dataset, architype, projection, prompt_agent, [client])
            scores.append((architype, score))

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

    for archetype, client in clients.items():
        print(f'Stopping {archetype}')
        await client.stop()

if __name__ == '__main__':
    asyncio.run(run_agents_benchmark())
