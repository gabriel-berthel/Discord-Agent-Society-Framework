# bench_runner.py

import asyncio
import pickle
import sys
import warnings

from benchmark.qa_tasks import *
from clients.prompt_client import PromptClient
from modules.agent_memories import Memories
from modules.agent_summuries import Contextualizer
from utils.agent_utils import *
from utils.base_prompt import generate_agent_prompt
from utils.file_utils import load_agent_logs, save_benchmark_results, load_yaml
import subprocess

warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")


def load_qa_bench_data() -> dict[str, SimpleNamespace]:
    archetypes = ["debunker", "nerd", "peacekeeper", "chameleon", "troll"]
    clients: dict[str, PromptClient] = PromptClient.build_clients('configs/qa_config.yaml')
    agent_logs: dict[str, SimpleNamespace] = {}

    with open('./output/qa_bench/qa_bench_histo.pkl', 'rb') as f:
        historic: list = pickle.load(f)

    for archetype in archetypes:
        personnality_prompt: str = generate_agent_prompt(archetype,
                                                         load_yaml('configs/archetypes.yaml')['agent_archetypes'][
                                                             archetype])
        agent_memories: Memories = \
            Memories(f'qa_bench_{archetype}_mem.pkl', 'output/qa_bench/memories').get_all_documents()[0]
        data: SimpleNamespace = load_agent_logs(f"output/qa_bench/logs/qa_bench_{archetype}_log.pkl")

        # Creates attributes such as logs.<archetype>.client 
        agent_logs[archetype] = SimpleNamespace(
            client=clients[archetype],
            personality=personnality_prompt,
            agent_memories=agent_memories,
            historic=historic,
            plans=[(x['input'], x['output']) for x in data.plans],
            reflections=[(x['input'], x['output']) for x in data.reflections],
            context_queries=[(x['input'], x['output']) for x in data.context_queries],
            neutral_ctxs=[(x['input'], x['output']) for x in data.neutral_ctxs],
            response_queries=[(x['input'], x['output']) for x in data.response_queries],
            memories=[(x['input'], x['output']) for x in data.memories],
            response=[(x['input'], x['output']) for x in data.response]
        )

    return agent_logs


async def run_benchmarks(archetype_logs: dict[str, SimpleNamespace]):
    benchmark_results = {
        # Recall / Probing
        'a1': {'description': 'Self-knowledge Recall,', 'archetypes': {}},
        'a2': {'description': 'Dialogue Recall', 'archetypes': {}},
        'a3': {'description': 'Reflection Recall', 'archetypes': {}},

        # Query Engine
        'b1': {'description': 'Context Queries Relevancy', 'archetypes': {}},
        'b2': {'description': 'Responder Queries Relevancy', 'archetypes': {}},

        # Contextualizer 
        'c1': {'description': 'Context Accuracy', 'archetypes': {}},

        # Models outputs
        'd1': {'description': 'Reflection Relevancy', 'archetypes': {}},
        'e1': {'description': 'Message Relevancy', 'archetypes': {}},
        'f1': {'description': 'Plan Relevancy', 'archetypes': {}}
    }

    for archetype, log in archetype_logs.items():
        memory = Memories(f'qa_bench_{archetype}_mem.pkl', 'output/qa_bench/memories')
        client = log.client

        print('Working on', archetype)

        await client.start()

        print('Starting A1')
        benchmark_results['a1']['archetypes'][archetype] = await run_a1(client, Prober(), log.personality)
        print('A1 DONE')

        print('Starting A2')
        benchmark_results['a2']['archetypes'][archetype] = await run_a2(client, Prober(), log.historic)
        print('A2 DONE')

        print('Starting A3')
        benchmark_results['a3']['archetypes'][archetype] = await run_a3(client, Prober(), log.agent_memories)
        print('A3 DONE')

        print('Starting B1')
        benchmark_results['b1']['archetypes'][archetype] = run_b1(log.context_queries, memory)
        print('B1 DONE')

        print('Starting B2')
        benchmark_results['b2']['archetypes'][archetype] = run_b2(log.response_queries, memory)
        print('B2 DONE')

        print('Starting C1')
        benchmark_results['c1']['archetypes'][archetype] = await run_c1(log.neutral_ctxs, Contextualizer('llama3:8b'))
        print('C1 DONE')

        print('Starting D1')
        benchmark_results['d1']['archetypes'][archetype] = run_d1(log.reflections, Prober())
        print('D1 DONE')

        print('Starting E1')
        benchmark_results['e1']['archetypes'][archetype] = run_e1(log.response, Prober())
        print('E1 DONE')

        print('Starting F1')
        benchmark_results['f1']['archetypes'][archetype] = run_f1(log.plans, Prober())
        print('F1 DONE')

        await client.stop()

        print(f'Saving results for {archetype}')
        save_benchmark_results(benchmark_results)

    return benchmark_results


if __name__ == "__main__":

    args = sys.argv[1:]

    should_run = 'n'
    run_time = 3600
    verbose = False

    for arg in args:
        if arg.lower() in ['y', 'n']:
            should_run = arg.lower()
        elif arg.isdigit():
            run_time = int(arg)
        elif arg.lower() == 'verbose':
            verbose = True

    if should_run == 'y':
        if verbose:
            print(f'About to run simulation for {run_time} seconds...')
        asyncio.run(PromptClient.prepare_qa_bench(run_time, verbose))

    print('Downloading en_core_web_sm')
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])

    logs = load_qa_bench_data()
    if verbose:
        print('Loaded logs, running benchmarks...')

    results = asyncio.run(run_benchmarks(logs))
