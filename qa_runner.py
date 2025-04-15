# bench_runner.py

import warnings
from types import SimpleNamespace
from qa_tasks import run_b1, run_b2, run_c1, run_d1
from modules.Memories import Memories
from modules.Contextualizer import Contextualizer
import pickle
from benchmark.Prober import Prober
from prompt_client import PromptClient
from utils.utils import *
from utils.prompt_generator import generate_agent_prompt

warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")

def load_logs(path):
    with open(path, "rb") as f:
        return SimpleNamespace(**pickle.load(f))

def load_qa_bench_data():
    archetypes = ["activist", "baseline", "fact_checker", "mediator", "trouble_maker"]
    logs = SimpleNamespace()

    with open('./qa_bench/qa_bench_histo.pkl', 'rb') as f:
        historic = pickle.load(f)
        
    clients = PromptClient.build_clients('qa_config.yaml')
    for archetype in archetypes:
        personnality_prompt, _ = generate_agent_prompt(archetype, load_yaml('archetypes.yaml')['agent_archetypes'][archetype])
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
            memories=[(x['input'], x['output']) for x in data.memories],
            summaries=[(x['input'], x['output']) for x in data.summuries],
            web_queries=[(x['input'], x['output']) for x in data.web_queries],
        ))
    return logs

async def run_a1(client: PromptClient, prober: Prober, personality):
    await client.start() 
    questions = prober.generate_sk_questions(personality, 10)
    responses = [await client.prompt(question, '100', 'Admin', 2) for question in questions]
    await client.stop()
    
    return Prober.evaluate(questions, responses)

async def run_a2(client: PromptClient, prober: Prober, dialogues):
    await client.start() 
    questions = prober.generate_content_questions("\n".join(dialogues), 25)
    responses = [await client.prompt(question, '100', 'Admin', 2) for question in questions]
    await client.stop()
    
    return Prober.evaluate(questions, responses)

async def run_a3(client: PromptClient, prober: Prober, reflections):
    await client.start() 
    questions = prober.generate_content_questions("\n".join(reflections), 25)
    responses = [await client.prompt(question, '100', 'Admin', 2) for question in questions]
    await client.stop()
    
    return Prober.evaluate(questions, responses)
import asyncio
def run_benchmarks(archetype_logs):

    
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

        results['a1']['archetypes'][archetype] = asyncio.run(run_a1(logs.client, Prober, logs.personality))
        results['a2']['archetypes'][archetype] = asyncio.run(run_a2(logs.client, Prober, logs.historic))
        results['a3']['archetypes'][archetype] = asyncio.run(run_a3(logs.client, Prober, logs.agent_memories))
        
        
        # results['b1']['archetypes'][archetype] = run_b1(logs.context_queries, memory)
        # results['b2']['archetypes'][archetype] = run_b2(logs.response_queries, memory)
        # results['c1']['archetypes'][archetype] = run_c1(logs.neutral_ctxs, Contextualizer('llama3.2'))
        # results['d1']['archetypes'][archetype] = run_d1(logs.reflections, Prober.classify_reflection_relevancy)
    return results

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
    results = run_benchmarks(archetype_logs)
    with open("a.txt", "w") as f:
        json.dump(results, f, indent=4, cls=NumpyEncoder)  
    print(results)
