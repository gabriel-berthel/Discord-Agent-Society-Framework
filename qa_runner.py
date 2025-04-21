# bench_runner.py

import warnings
from types import SimpleNamespace
from qa_tasks import run_b1, run_b2, run_c1, run_d1, run_a1, run_a2, run_a3, run_e1, run_f1
from modules.Memories import Memories
from modules.Contextualizer import Contextualizer
import pickle
from utils.utils import *
from utils.prompt_generator import generate_agent_prompt
from utils.Prober import Prober
from prompt_client import PromptClient
import asyncio
import os

warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")

async def prepare_qa_bench():
    import os
    import pickle
    import shutil
    import shutil

    shutil.rmtree('output/qa_bench')
    os.makedirs('output/qa_bench')
    os.makedirs('output/qa_bench/logs')
    os.makedirs('output/qa_bench/memories')
    
    print_replies = True
    simulation_duration = 60 * 60
    clients, historic = await PromptClient.run_simulation(simulation_duration, print_replies, config_file='configs/qa_bench_prepare.yaml')
    
    for archetype, client in clients.items():
        await client.stop()
        client.agent.logger.save_logs()

    file_path = os.path.join(f"qa_bench/qa_bench_histo.pkl")

    with open(file_path, "wb") as f:
        pickle.dump(historic, f)

    print(f"[LOG] Saved historic to {file_path}")   

def load_logs(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
        return SimpleNamespace(**obj)

def load_qa_bench_data():
    archetypes = ["debunker", "nerd", "peacekeeper", "chameleon", "troll"]
    clients = PromptClient.build_clients('configs/qa_config.yaml')
    logs = {}

    with open('./qa_bench/qa_bench_histo.pkl', 'rb') as f:
        historic = pickle.load(f)
        
    for archetype in archetypes:
        
        personnality_prompt = generate_agent_prompt(archetype, load_yaml('archetypes.yaml')['agent_archetypes'][archetype])
        agent_memories = Memories(f'qa_bench_{archetype}_mem.pkl', 'qa_bench/memories').get_all_documents()[0]
        data = load_logs(f"qa_bench/logs/qa_bench_{archetype}_log.pkl")
        
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

async def run_benchmarks(logs):
    
    if os.path.exists("results.json"):
        with open("results.json", "r") as file:
            results = json.load(file)
    
    else:        
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
        memory = Memories(f'qa_bench_{archetype}_mem.pkl', 'qa_bench/memories')
        print('Working on', archetype)
        await log.client.start()

        print('Starting')
        results['a1']['archetypes'][archetype] = await run_a1(log.client, Prober, log.personality)
        print('A1 DONE')
        results['a2']['archetypes'][archetype] = await run_a2(log.client, Prober, log.historic)
        print('A2 DONE')
        results['a3']['archetypes'][archetype] = await run_a3(log.client, Prober, log.agent_memories)
        print('A3 DONE')
        results['b1']['archetypes'][archetype] = run_b1(log.context_queries, memory)
        print('B1 DONE')
        save_results(results)
        results['b2']['archetypes'][archetype] = run_b2(log.response_queries, memory)
        print('B2 DONE')
        save_results(results)
        results['c1']['archetypes'][archetype] = await run_c1(log.neutral_ctxs, Contextualizer('llama3:8b'))
        print('C1 DONE')
        save_results(results)
        results['d1']['archetypes'][archetype] = run_d1(log.reflections, Prober)
        print('D1 DONE')
        save_results(results)
        results['e1']['archetypes'][archetype] = run_e1(log.response, Prober)
        print('E1 DONE')
        results['f1']['archetypes'][archetype] = run_f1(log.plans, Prober)
        save_results(results)
        print('F1 DONE')
        save_results(results)
        results['e1']['archetypes'][archetype] = run_e1(log.response, Prober)
        print('E1 DONE')
        save_results(results)

        await logs.client.stop()
    return results

def save_results(results):
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4, cls=NumpyEncoder) 

if __name__ == "__main__":
    
    import json
    import numpy as np
      
    asyncio.run(prepare_qa_bench())
      
    logs = load_qa_bench_data()
    results = asyncio.run(run_benchmarks(logs))
    