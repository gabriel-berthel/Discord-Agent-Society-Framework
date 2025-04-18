# bench_runner.py

import warnings
from types import SimpleNamespace
from qa_tasks import run_b1, run_b2, run_c1, run_d1, run_a1, run_a2, run_a3
from modules.Memories import Memories
from modules.Contextualizer import Contextualizer
import pickle
from utils.utils import *
from utils.prompt_generator import generate_agent_prompt
from benchmark.Prober import Prober
from prompt_client import PromptClient

warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")

def load_logs(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
        return SimpleNamespace(**obj)

def load_qa_bench_data():
    archetypes = ["debunker", "nerd", "peacekeeper", "chameleon", "troll"]
    logs = SimpleNamespace()

    with open('./qa_bench/qa_bench_histo.pkl', 'rb') as f:
        historic = pickle.load(f)
        
    clients = PromptClient.build_clients('configs/qa_config.yaml')
    for archetype in archetypes:
        personnality_prompt = generate_agent_prompt(archetype, load_yaml('archetypes.yaml')['agent_archetypes'][archetype])
        agent_memories = Memories(f'qa_bench_{archetype}_mem.pkl', 'qa_bench/memories').get_all_documents()[0]
        data = load_logs(f"qa_bench/logs/qa_bench_{archetype}_log.pkl")
        
        setattr(logs, archetype, SimpleNamespace(
            client=clients[archetype],
            personality=personnality_prompt,
            agent_memories=agent_memories,
            historic=historic,
            plans=[(x['input'], x['output']) for x in data.plans],
            reflections=[(x['input'], x['output']) for x in data.reflections],
            context_queries=[(x['input'], x['output']) for x in data.context_queries],
            neutral_ctxs=[(x['input'], x['output']) for x in data.neutral_ctxs],
            response_queries=[(x['input'], x['output']) for x in data.response_queries],
            memories=[(x['input'], x['output']) for x in data.memories]
        ))
    return logs

import asyncio
async def run_benchmarks(archetype_logs):

    results = {
        'a1': {'description': 'Self-knowledge Recall,', 'archetypes': {}},
        'a2': {'description': 'Dialogue Recall', 'archetypes': {}},
        'a3': {'description': 'Reflection Recall', 'archetypes': {}},
        'b1': {'description': 'Context Queries Relevancy', 'archetypes': {}},
        'b2': {'description': 'Responder Queries Relevancy', 'archetypes': {}},
        'c1': {'description': 'Context Accuracy', 'archetypes': {}},
        'd1': {'description': 'Reflection Relevancy', 'archetypes': {}}
    }

    for archetype, logs in archetype_logs:
        memory = Memories(f'qa_bench_{archetype}_mem.pkl', 'qa_bench/memories')
        print('Working on', archetype)
        await logs.client.start()
        results['a1']['archetypes'][archetype] = await run_a1(logs.client, Prober, logs.personality)
        print('A1 DONE')
        save_results(results)
        results['b1']['archetypes'][archetype] = run_b1(logs.context_queries, memory)
        print('B1 DONE')
        save_results(results)
        results['b2']['archetypes'][archetype] = run_b2(logs.response_queries, memory)
        print('B2 DONE')
        save_results(results)
        results['c1']['archetypes'][archetype] = await run_c1(logs.neutral_ctxs, Contextualizer('llama3:8b'))
        print('C1 DONE')
        save_results(results)
        results['d1']['archetypes'][archetype] = run_d1(logs.reflections, Prober.classify_reflection_relevancy)
        print('D1 DONE')
        save_results(results)
        results['a2']['archetypes'][archetype] = await run_a2(logs.client, Prober, logs.historic)
        print('A2 DONE')
        save_results(results)
        results['a3']['archetypes'][archetype] = await run_a3(logs.client, Prober, logs.agent_memories)
        print('A3 DONE')
        save_results(results)
        await logs.client.stop()
    return results

def save_results(results):
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4, cls=NumpyEncoder) 

if __name__ == "__main__":
    
    import json
    import numpy as np
    
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, np.generic):
                return obj.item()
            return super().default(obj)
        
    import json
    logs = load_qa_bench_data()
    archetype_logs = [(role, getattr(logs, role)) for role in vars(logs)]
    results = asyncio.run(run_benchmarks(archetype_logs))
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4, cls=NumpyEncoder)  
    print(results)
