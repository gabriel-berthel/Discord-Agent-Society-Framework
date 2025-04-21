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
from utils.Prober import Prober
from prompt_client import PromptClient
from modules.Memories import Memories
from modules.Contextualizer import Contextualizer

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

def run_b1(context_queries, memory_module: Memories):
    cos_baselines = []
    cos_agents = []
    for msgs, queries in context_queries:
        agent = memory_module.query_multiple(queries)
        baseline = memory_module.query_multiple(list(msgs))
        cos_baselines.append(np.mean(compute_cosine_distances(baseline)))
        cos_agents.append(np.mean(compute_cosine_distances(agent)))
    
    return {
        'std': {'baseline': np.std(cos_baselines), 'agent': np.std(cos_agents)},
        'mean': {'baseline': np.mean(cos_baselines), 'agent': np.mean(cos_agents)},
        'min': {'baseline': np.min(cos_baselines), 'agent': np.min(cos_agents)},
        'max': {'baseline': np.max(cos_baselines), 'agent': np.max(cos_agents)}
    }

def run_b2(logs, memory_module: Memories):
    cos_baselines = []
    cos_agents = []
    for arguments, queries in logs:
        plan, context, personnality_prompt, messages = arguments
        agent = memory_module.query_multiple(queries)
        baseline = memory_module.query_multiple([plan, context, personnality_prompt] + messages)
        cos_baselines.append(np.mean(compute_cosine_distances(baseline)))
        cos_agents.append(np.mean(compute_cosine_distances(agent)))

    return {
        'std': {'baseline': np.std(cos_baselines), 'agent': np.std(cos_agents)},
        'mean': {'baseline': np.mean(cos_baselines), 'agent': np.mean(cos_agents)},
        'min': {'baseline': np.min(cos_baselines), 'agent': np.min(cos_agents)},
        'max': {'baseline': np.max(cos_baselines), 'agent': np.max(cos_agents)}
    }

async def run_c1(neutral_ctxs, contextualizer: Contextualizer):
    shared_keywords, unique_keywords  = [], []
    cosine_similarities_baselines = []
    cosine_similarities_agents = []
    
    for arguments, _ in neutral_ctxs:
        msgs, bot_context = arguments
        content = bot_context + '\n'.join(msgs)
        
        agent = await contextualizer.neutral_context(msgs, bot_context)
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

def run_d1(reflections_logs, prober: Prober):
    print('Running D1')
    results = []
    for arguments, reflection in reflections_logs:
        messages, personality_prompt = arguments
        
        axes = {
            "personality": personality_prompt,
            "dialogue": '\n'.join(messages)
        }

        results.append(
            prober.make_scaled_relevancy_poll("personnal reflection", axes, reflection)
        )

    print('Evaluating D1')
    return prober.evaluate_scales(results)


def run_e1(response_log, prober: Prober):
    print('Running E1')
    results = []
    for arguments, response in response_log:
        plan, context, memories, messages, personality = arguments
        
        axes = {
            "personality": personality,
            "plan": plan,
            "memories": '\n'.join(memories),
            "context": context,
            "dialogue": '\n'.join(messages)
        }

        results.append(
            prober.make_scaled_relevancy_poll("discord response", axes, response)
        )

    print('Evaluating E1')
    return prober.evaluate_scales(results)


def run_f1(plan_log, prober:Prober):
    print('Running F1')
    results = []
    for arguments, new_plan in plan_log:
        former_plan, context, memories, personality = arguments
        
        axes = {
            "personality": personality,
            "memories": '\n'.join(memories),
            "context": context,
            "former_plan": former_plan
        }

        results.append(
            prober.make_scaled_relevancy_poll(
                "new plan", 
                axes, 
                new_plan
            )
        )


    print('Evaluating F1')
    return prober.evaluate_scales(results)

def force_ctx(self):
    return f"You must absolutely answer to ADMIN as demanded and only as demanded."

async def run_a1(client: PromptClient, prober: Prober, personality):
    client.get_bot_context = force_ctx.__get__(client)
    questions = [q for q in prober.generate_content_questions(personality, 12)]
    print(f'Probing agent: {client.name}')
    responses = [await client.prompt(question['question'], 100, 'ADMIN', 1, f'a1_{client.name}_{client.agent.archetype}.txt') for question in questions]
    return Prober.evaluate_qa(questions, responses)

def chunk_list(lst, chunk_size):
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

async def run_a2(client: PromptClient, prober: Prober, dialogues):
    client.get_bot_context = force_ctx.__get__(client)
    questions = []
    for i, chunk in enumerate(chunk_list(dialogues, 10)):
        print(f'Chunk {i+1}')
        questions.extend(prober.generate_content_questions("\n".join(chunk), 4))
    
    print(f'Probing agent: {client.name}')
    responses = [await client.prompt(question['question'], 100, 'ADMIN', 1, f'a2_{client.name}_{client.agent.archetype}.txt') for question in questions]
    
    return Prober.evaluate_qa(questions, responses)

async def run_a3(client: PromptClient, prober: Prober, reflections):
    client.get_bot_context = force_ctx.__get__(client)
    questions = []
    for i, chunk in enumerate(chunk_list(reflections, 5)):
        print(f'Chunk {i+1}')
        questions.extend(prober.generate_content_questions("\n".join(chunk), 4))
    
    print(f'Probing agent: {client.name}')
    responses = [await client.prompt(question['question'], 100, 'ADMIN', 1, f'a3_{client.name}_{client.agent.archetype}.txt') for question in questions]
    
    return Prober.evaluate_qa(questions, responses)
