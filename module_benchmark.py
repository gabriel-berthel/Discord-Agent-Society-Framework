import pickle
from types import SimpleNamespace
import asyncio
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances
model = SentenceTransformer('all-MiniLM-L6-v2')
import numpy as np

def load_logs(path):
    with open(path, "rb") as f:
        return SimpleNamespace(**pickle.load(f))

def load_qa_bench_data():
    roles = ["activist", "baseline", "fact_checker", "mediator", "trouble_maker"]
    logs = SimpleNamespace()

    for role in roles:
        data = load_logs(f"qa_bench/qa_bench_{role}.pkl")
        role_logs = SimpleNamespace(
            plans=[(item['input'], item['output']) for item in data.plans],
            reflections=[(item['input'], item['output']) for item in data.reflections],
            context_queries=[(item['input'], item['output']) for item in data.context_queries],
            neutral_ctxs=[(item['input'], item['output']) for item in data.neutral_ctxs],
            response_queries=[(item['input'], item['output']) for item in data.response_queries],
            memories=[(item['input'], item['output']) for item in data.memories],
            summaries=[(item['input'], item['output']) for item in data.summuries],
            web_queries=[(item['input'], item['output']) for item in data.web_queries]
        )
        setattr(logs, role, role_logs)

    return logs

# Load dataset
logs = load_qa_bench_data()

# Access logs by role
archetype_logs = [
    ("activist", logs.activist),
    ("baseline", logs.baseline),
    ("fact_checker", logs.fact_checker),
    ("mediator", logs.mediator),
    ("trouble_maker", logs.trouble_maker)
]

from modules.Memories import Memories
from modules.QueryEngine import QueryEngine


def compute_cosine_distances(documents_x, documents_y, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings_x = model.encode(documents_x)
    embeddings_y = model.encode(documents_y)
    cosine_distances_matrix = cosine_distances(embeddings_x, embeddings_y)
    average_cosine_distance = np.mean(cosine_distances_matrix)
    std_cosine_distance = np.std(cosine_distances_matrix)
    return average_cosine_distance, std_cosine_distance

def query_engine_effectiveness(archetype_logs):
    
    for archetype, logs in archetype_logs:
        memory_module = Memories(f'qa_bench_{archetype}_mem.pkl', 'qa_bench/memories')

        for memories, response_queries in zip(logs.memories, logs.response_queries):
            
            arguments, queries = response_queries
            b1_baseline = response_queries.query_multiple(list(arguments))
            b1_agent = response_queries.query_multiple(queries)
            
            avg, std = compute_cosine_distances(b1_baseline, b1_agent)
            
            #  self.logs['response_queries'].append({'input': (plan, context, messages), 'output': queries})
            print(queries)
            # res = asyncio.create_task(QueryEngine('llama3.2').context_query())
            # print(res)

query_engine_effectiveness(archetype_logs)
        
