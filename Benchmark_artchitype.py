import asyncio
import json
import time
from datetime import datetime
from promptbench import load_benchmark
from prompt_client import PromptClient
from modules.DiscordServer import DiscordServer

ARCHETYPES = ['baseline', 'trouble_maker', 'fact_checker', 'activist', 'moderator']

async def run_agents_benchmark():
    server = DiscordServer(1, 'Benchmarking', 1)
    server.update_user(1, 'User')
    server.add_channel(1, 'General')

    results = []
    tasks = ["commonsense_reasoning", "logical_reasoning", "world_knowledge"]

    for archetype in ARCHETYPES:
        print(f"\nBenchmarking archetype: {archetype}")
        agent = PromptClient('benchmark_config.yaml', archetype, f"{archetype}_agent", 1, server)

        # Override get_bot_context pour forcer la réponse
        def forced_context():
            return "Répondez toujours quoi qu'il arrive."

        agent.agent.get_bot_context = forced_context

        await agent.start()

        for task in tasks:
            print(f"Task: {task}")
            dataset = load_benchmark(task)

            for sample in dataset:
                prompt = sample["input"]
                server.add_message(1, 1, 'User', prompt)

                start_time = time.time()
                response = await agent.prompt(prompt, 1, 'User')
                end_time = time.time()

                results.append({
                    "task": task,
                    "archetype": archetype,
                    "input": prompt,
                    "response": response,
                    "response_time": end_time - start_time,
                    "response_length": len(response.split())
                })

                await asyncio.sleep(0.5)

    filename = f"agents_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Benchmark des agents terminé. Résultats sauvegardés dans '{filename}'.")

if __name__ == '__main__':
    asyncio.run(run_agents_benchmark())
