# bench_runner.py

import warnings
from types import SimpleNamespace
from qa_tasks import run_b1, run_b2, run_c1
from modules.Memories import Memories
from modules.Contextualizer import Contextualizer
import pickle

warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub.file_download")

def load_logs(path):
    with open(path, "rb") as f:
        return SimpleNamespace(**pickle.load(f))

def load_qa_bench_data():
    roles = ["activist", "baseline", "fact_checker", "mediator", "trouble_maker"]
    logs = SimpleNamespace()

    for role in roles:
        data = load_logs(f"qa_bench/logs/qa_bench_{role}_log.pkl")
        setattr(logs, role, SimpleNamespace(
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

def run_benchmarks(archetype_logs):
    results = {
        'b1': {'description': 'Context Queries Relevancy', 'archetypes': {}},
        'b2': {'description': 'Responder Queries Relevancy', 'archetypes': {}},
        'c1': {'description': 'Context Accuracy', 'archetypes': {}}
    }

    for archetype, logs in archetype_logs:
        memory = Memories(f'qa_bench_{archetype}_mem.pkl', 'qa_bench/memories')
        # results['b1']['archetypes'][archetype] = run_b1(logs.context_queries, memory)
        # results['b2']['archetypes'][archetype] = run_b2(logs.response_queries, memory)
        results['c1']['archetypes'][archetype] = run_c1(logs.neutral_ctxs, Contextualizer('llama3.2'))

    return results

if __name__ == "__main__":
    logs = load_qa_bench_data()
    archetype_logs = [(role, getattr(logs, role)) for role in vars(logs)]
    results = run_benchmarks(archetype_logs)
    print(results)
