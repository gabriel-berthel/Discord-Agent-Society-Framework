import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from sentence_transformers import SentenceTransformer

def compute_cosine_distances(documents, model_name='all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(documents, convert_to_numpy=True)
    return cosine_distances(embeddings)

def run_b1(context_queries, memory_module):
    cos_baselines = []
    cos_agents = []
    for msgs, queries in context_queries:
        agent = memory_module.query_multiple(queries)
        baseline = memory_module.query_multiple(list(msgs))
        cos_baselines.append(np.mean(compute_cosine_distances(baseline)))
        cos_agents.append(np.mean(compute_cosine_distances(agent)))
    
    return {
        'task': 'b1',
        'name': 'Neutral Queries Relevancy',
        'std': {'baseline': np.std(cos_baselines), 'agent': np.std(cos_agents)},
        'mean': {'baseline': np.mean(cos_baselines), 'agent': np.mean(cos_agents)}
    }

def run_b2(logs, memory_module):
    cos_baselines = []
    cos_agents = []
    for arguments, queries in logs:
        plan, context, messages = arguments
        agent = memory_module.query_multiple(queries)
        baseline = memory_module.query_multiple([plan, context] + messages)
        cos_baselines.append(np.mean(compute_cosine_distances(baseline)))
        cos_agents.append(np.mean(compute_cosine_distances(agent)))

    return {
        'task': 'b2',
        'name': 'Responder Queries Relevancy',
        'std': {'baseline': np.std(cos_baselines), 'agent': np.std(cos_agents)},
        'mean': {'baseline': np.mean(cos_baselines), 'agent': np.mean(cos_agents)}
    }
