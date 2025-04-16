import numpy as np
from sklearn.metrics.pairwise import cosine_distances
from sentence_transformers import SentenceTransformer
from transformers import BartTokenizer, BartForConditionalGeneration
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch
from utils.utils import list_to_text
import spacy
import asyncio
import pytextrank
from benchmark.Prober import Prober
from prompt_client import PromptClient

model = None
tokenizer = None
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("textrank")

def extract_keywords(text, num_keywords=10):
    doc = nlp(text)
    keywords = [phrase.text.lower() for phrase in doc._.phrases[:num_keywords]]
    return set(keywords)

def compare_embeddings(agent_text, baseline_text):
    documents = [agent_text, baseline_text]
    distance_matrix = compute_cosine_distances(documents)
    return 1 - distance_matrix[0][1]

def compare_keywords(agent_keywords, baseline_keywords):
    common_keywords = agent_keywords.intersection(baseline_keywords)
    return len(common_keywords), len(agent_keywords.union(baseline_keywords))

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
    shared_keywords, unique_keywords  = [], []
    cosine_similarities_baselines = []
    cosine_similarities_agents = []
    
    print(len(neutral_ctxs))
    for arguments, _ in neutral_ctxs:
        msgs, bot_context = arguments
        content = bot_context + '\n'.join(msgs)
        
        agent = asyncio.run(contextualizer.neutral_context(msgs, bot_context))
        baseline = summarize_text(content)
        
        agent_keywords = extract_keywords(agent)
        baseline_keywords = extract_keywords(baseline)
        
        shared, unique = compare_keywords(agent_keywords, baseline_keywords)
        
        shared_keywords.append(shared)
        unique_keywords.append(unique)
        
        cosine_similarity_baseline = compute_cosine_distances([baseline, content])
        cosine_similarity_agent = compute_cosine_distances([agent, content])

        cosine_similarities_baselines.append(cosine_similarity_baseline)
        cosine_similarities_agents.append(cosine_similarity_agent)
    
    shared_mean = np.array(shared_keywords).mean()
    unique_mean = np.array(unique_keywords).mean()
    ratio = shared_mean / unique_mean if unique_mean != 0 else float('inf')
    
    return {
        'shared_keyword_ratio': ratio,
        'cosine_similarity': {
            'baseline': np.mean(cosine_similarities_baselines),
            'agent': np.mean(cosine_similarities_agents)
        }
    }

def run_d1(reflections_logs, evaluator_fn):
    total_scores = {"personality": 0, "dialogue": 0}
    num_reflections = 0

    for arguments, reflection in reflections_logs:
        messages, _, personality_prompt = arguments
        result = evaluator_fn('\n'.join(messages), reflection, personality_prompt)
        scores = result["relevancy_scores"]

        for key in total_scores:
            total_scores[key] += scores.get(key, 0)

        num_reflections += 1

    final_avg_scores = {k: v / num_reflections for k, v in total_scores.items()}
    overall_average_score = sum(final_avg_scores.values()) / len(final_avg_scores)

    return {
        "total_scores": total_scores,
        "average_scores": final_avg_scores,
        "overall_average_score": overall_average_score
    }

async def run_a1(client: PromptClient, prober: Prober, personality):
    questions = prober.generate_sk_questions(personality, 10)
    responses = [await client.prompt(question['question'], 100, 'Admin', 1) for question in questions]
    return Prober.evaluate(questions, responses)

def chunk_list(lst, chunk_size):
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

async def run_a2(client: PromptClient, prober: Prober, dialogues):
    chunks = chunk_list(dialogues, 5)
    
    questions = [
        question
        for chunk in chunks[0]
        for question in prober.generate_content_questions("\n".join(chunk), 4)
    ]
    
    print(questions)
    responses = [await client.prompt(question['question'], 100, 'Admin', 1) for question in questions]
    
    return Prober.evaluate(questions, responses)

async def run_a3(client: PromptClient, prober: Prober, reflections):
    chunks = chunk_list(reflections, 6)
    
    questions = []
    for chunk in chunks:
        questions.append(prober.generate_content_questions("\n".join(chunk), 5))
    
    questions = prober.generate_content_questions("\n".join(reflections), 25)
    responses = [await client.prompt(question['question'], 100, 'Admin', 1) for question in questions]
    
    return Prober.evaluate(questions, responses)
