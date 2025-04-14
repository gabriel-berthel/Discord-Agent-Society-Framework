import promptbench as pb
import json
from datetime import datetime
from tqdm import tqdm
import prompt_client
import asyncio
import ollama

async def prompt_ollama(prompt):
    
    return ollama.generate("llama3.2", prompt)["response"]

async def prompt_agent(prompt, client): 
    
    return await client.prompt(prompt, 2, "moderateur")

RESULTS = []
clients = prompt_client.PromptClient.build_clients()
def get_projection_fn():
    return lambda pred: 1 if "positive" in pred.lower() else 0 if "negative" in pred.lower() else -1

tasks = [
    ("sentiment", pb.Prompt([f"Classify the sentence as positive or negative: {{content}}"]), get_projection_fn, "sst2"),
    
]

async def run_task(prompts, dataset, architype, projection, prompt_fn, args = []):
     for prompt in prompts:
        preds, labels = [], []
        for data in tqdm(dataset, desc=f"{architype} - {dataset}"):
            input_text = pb.InputProcess.basic_format(prompt, data)
            label = data['label']
            raw_pred = await prompt_fn(input_text, *args)
            pred = pb.OutputProcess.cls(raw_pred, projection)
            preds.append(pred, label)
        # evaluate
        return pb.Eval.compute_cls_accuracy(preds, labels) 

async def run_agents_benchmark():

    for task, prompts, projection, dataset in tasks :

        dataset = pb.DatasetLoader.load_dataset(dataset)
        scores = []
        for architype, client in clients.items():
            await client.start() 
            score = await run_task(prompts, dataset, architype, projection, prompt_agent, [client])
            scores.append((architype, score))
            await client.stop()
        baseline_score = await run_task(prompts, dataset, architype, projection, prompt_ollama)

        RESULTS.append({
        "dataset": dataset,
        "scores": scores,
        "task": task,
        "baseline": baseline_score
        })

    

if __name__ == '__main__':
    asyncio.run(run_agents_benchmark())
