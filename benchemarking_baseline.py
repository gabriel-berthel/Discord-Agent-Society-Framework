import json
import asyncio
from datetime import datetime
from promptbench import load_benchmark
import ollama
import time

MODEL = "llama3.2"

async def query_ollama(prompt):
    start_time = time.time()
    response = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}])
    end_time = time.time()
    return response['message']['content'].strip(), end_time - start_time

async def run_baseline():
    results = []
    tasks = ["commonsense_reasoning", "logical_reasoning", "world_knowledge"]

    for task in tasks:
        print(f"\nBenchmarking task: {task}")
        dataset = load_benchmark(task)

        for sample in dataset:
            prompt = sample["input"]
            response, response_time = await query_ollama(prompt)

            results.append({
                "task": task,
                "archetype": "baseline",
                "input": prompt,
                "response": response,
                "response_time": response_time,
                "response_length": len(response.split())
            })

    filename = f"baseline_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n Baseline benchmark terminé. Résultats sauvegardés dans '{filename}'.")

if __name__ == '__main__':
    asyncio.run(run_baseline())
