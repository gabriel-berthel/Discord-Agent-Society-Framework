# bench_runner.py

import warnings
from types import SimpleNamespace
from benchmark.qa_tasks import *
from modules.agent_memories import Memories
from modules.agent_summuries import Contextualizer
import pickle
from utils.agent_utils import *
from utils.base_prompt import generate_agent_prompt
from benchmark.agent_prober import Prober
from clients.prompt_client import PromptClient
import asyncio
import os
import shutil
from utils.qa_utils import *

warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")

async def prepare_qa_bench(duration, print_replies, config_file):
    shutil.rmtree('output/qa_bench')
    os.makedirs('output/qa_bench')
    os.makedirs('output/qa_bench/logs')
    os.makedirs('output/qa_bench/memories')
    
    print('Running simulatation')
    clients, historic = await PromptClient.run_simulation(
        duration, print_replies, config_file='configs/qa_bench_prepare.yaml'
    )
    
    print('Saving logs in output/qa_bench/logs')
    for archetype, client in clients.items():
        await client.stop()
        client.agent.logger.save_logs()

    print('Saving historic in output/qa_bench/')
    with open("output/qa_bench/qa_bench_histo.pkl", "wb") as f:
        pickle.dump(historic, f)   
    with open("output/qa_bench/qa_bench_histo.txt", "w") as f:
        for line in historic:
            f.writelines(f'{line}\n')

def load_qa_bench_data():
    archetypes = ["debunker", "nerd", "peacekeeper", "chameleon", "troll"]
    clients: PromptClient = PromptClient.build_clients('configs/qa_config.yaml')
    logs:dict[str, SimpleNamespace] = {}

    with open('./output/qa_bench/qa_bench_histo.pkl', 'rb') as f:
        historic: list = pickle.load(f)
        
    for archetype in archetypes:
        
        personnality_prompt: str = generate_agent_prompt(archetype, load_yaml('archetypes.yaml')['agent_archetypes'][archetype])
        agent_memories: Memories = Memories(f'qa_bench_{archetype}_mem.pkl', 'output/qa_bench/memories').get_all_documents()[0]
        data: SimpleNamespace = load_logs(f"output/qa_bench/logs/qa_bench_{archetype}_log.pkl")
        
        # Creates attributes such as logs.<archetype>.client 
        logs[archetype] = SimpleNamespace(
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
        
    return logs

async def run_benchmarks(logs: dict):
    
    results = {
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

    for archetype, log in logs.items():
        memory = Memories(f'qa_bench_{archetype}_mem.pkl', 'output/qa_bench/memories')
        
        print('Working on', archetype)
        
        await log.client.start()

        print('Starting A1')
        results['a1']['archetypes'][archetype] = await run_a1(log.client, Prober, log.personality)
        print('A1 DONE')
        
        print('Starting A2')
        results['a2']['archetypes'][archetype] = await run_a2(log.client, Prober, log.historic)
        print('A2 DONE')
        
        print('Starting A3')
        results['a3']['archetypes'][archetype] = await run_a3(log.client, Prober, log.agent_memories)
        print('A3 DONE')
        
        print('Starting B1')
        results['b1']['archetypes'][archetype] = run_b1(log.context_queries, memory)
        print('B1 DONE')
        
        print('Starting B2')
        results['b2']['archetypes'][archetype] = run_b2(log.response_queries, memory)
        print('B2 DONE')
        
        print('Starting C1')
        results['c1']['archetypes'][archetype] = await run_c1(log.neutral_ctxs, Contextualizer('llama3:8b'))
        print('C1 DONE')
        
        print('Starting D1')
        results['d1']['archetypes'][archetype] = run_d1(log.reflections, Prober)
        print('D1 DONE')

        print('Starting E1')
        results['e1']['archetypes'][archetype] = run_e1(log.response, Prober)
        print('E1 DONE')
        
        print('Starting F1')
        results['f1']['archetypes'][archetype] = run_f1(log.plans, Prober)
        print('F1 DONE')
        
        await logs.client.stop()
        
        print(f'Saving results for {archetype}')
        save_results()

    return results

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
        asyncio.run(prepare_qa_bench(run_time, verbose, 'configs/qa_bench_prepare.yaml'))
    
    logs = load_qa_bench_data()
    if verbose:
        print('Loaded logs, running benchmarks...')
        
    results = asyncio.run(run_benchmarks(logs))

    