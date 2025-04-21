import spacy
import pytextrank
from sentence_transformers import SentenceTransformer
from transformers import BartTokenizer, BartForConditionalGeneration
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics.pairwise import cosine_similarity
import torch
import json
import numpy as np
import pickle
from types import SimpleNamespace

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

# -----

def load_logs(path):
    with open(path, "rb") as f:
        obj = pickle.load(f)
        return SimpleNamespace(**obj)

def save_results(results):
    with open("outputs/qa_bench/results.json", "w") as f:
        json.dump(results, f, indent=4, cls=NumpyEncoder) 

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
