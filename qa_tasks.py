import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from sentence_transformers import SentenceTransformer
from transformers import BartTokenizer, BartForConditionalGeneration
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from utils.utils import list_to_text
import spacy
import pytextrank

model = None
tokenizer = None
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("textrank")

def extract_keywords(text, num_keywords=10):
    doc = nlp(text)
    keywords = set()
    for phrase in doc._.phrases[:num_keywords]:
        keywords.add(phrase.text.lower())
    return keywords

def compare_embeddings(agent_text, baseline_text):
    documents = [agent_text, baseline_text]
    distance_matrix = compute_cosine_distances(documents)
    return 1 - distance_matrix[0][1]

def compare_keywords(agent_keywords, baseline_keywords):
    common_keywords = agent_keywords.intersection(baseline_keywords)
    return len(common_keywords)

def summarize_text(text):
    global model, tokenizer
    
    if model is None or tokenizer is None:
        model_name = "facebook/bart-large-cnn"
        tokenizer = BartTokenizer.from_pretrained(model_name)
        model = BartForConditionalGeneration.from_pretrained(model_name)

    if not text.strip():
        return "Input text is empty."

    inputs = tokenizer([text], max_length=1024, return_tensors="pt", truncation=True)
    summary_ids = model.generate(
        inputs["input_ids"],
        max_length=1024,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True,
    )
    
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

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
        'std': {'baseline': np.std(cos_baselines), 'agent': np.std(cos_agents)},
        'mean': {'baseline': np.mean(cos_baselines), 'agent': np.mean(cos_agents)}
    }

def run_c1(neutral_ctxs, contextualizer):
    keyword_counts_baselines = []
    keyword_counts_agents = []
    cosine_similarities_baselines = []
    cosine_similarities_agents = []
    
    for arguments, _ in neutral_ctxs:
        msgs, bot_context = arguments
        agent = contextualizer.neutral_context(msgs, bot_context)
        baseline = summarize_text('\n'.join(msgs))
        
        agent_keywords = extract_keywords(agent)
        baseline_keywords = extract_keywords(baseline)
        
        keyword_count = compare_keywords(agent_keywords, baseline_keywords)
        
        cosine_similarity_baseline = compute_cosine_distances([baseline])
        cosine_similarity_agent = compute_cosine_distances([agent])
        
        keyword_counts_baselines.append(keyword_count)
        keyword_counts_agents.append(keyword_count)
        cosine_similarities_baselines.append(cosine_similarity_baseline)
        cosine_similarities_agents.append(cosine_similarity_agent)
    
    return {
        'keyword_count': {
            'baseline': keyword_counts_baselines,
            'agent': keyword_counts_agents
        },
        'cosine_similarity': {
            'baseline': cosine_similarities_baselines,
            'agent': cosine_similarities_agents
        }
    }
