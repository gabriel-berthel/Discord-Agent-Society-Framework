import spacy
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_distances
from transformers import BartTokenizer, BartForConditionalGeneration

# it is not "called" anywhere but nlp.add_pipe("textrank") depends on it :)
# noinspection PyUnresolvedReferences
import pytextrank

model = None
tokenizer = None
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("textrank")


def chunk_list(lst, chunk_size):
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


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
    global model
    model = SentenceTransformer(model_name)
    embeddings = model.encode(documents, convert_to_numpy=True)
    return cosine_distances(embeddings)
