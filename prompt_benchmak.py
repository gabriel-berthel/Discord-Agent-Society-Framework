import json
import time
import requests
from datetime import datetime
from pathlib import Path
import ollama

# ====== Config ======
PROMPTS_FILE = 'prompts.jsonl'
RESULTS_FILE = 'results.jsonl'
DISCORD_WEBHOOK_URL = 'https://discord.com/api/webhooks/TON_WEBHOOK_ICI'  # Remplace par ton webhook
MODELS = ['base_model', 'archetype_1', 'archetype_2', 'archetype_3', 'archetype_4', 'archetype_5']

# Contrainte de délai de config = -1
OLLAMA_OPTIONS = {
    'num_predict': -1,
    'temperature': 0.7,
}

# Freeze mémoire pour éviter les fuites
import gc
import tracemalloc
tracemalloc.start()

# ====== Fonctions utiles ======

def load_prompts(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f if line.strip()]

def save_result(result, file_path):
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(result, ensure_ascii=False) + '\n')

def send_to_discord(content):
    data = {"content": content}
    try:
        response = requests.post(DISCORD_WEBHOOK_URL, json=data)
        if response.status_code != 204:
            print(f"Erreur Discord: {response.text}")
    except Exception as e:
        print(f"Erreur lors de l'envoi Discord: {e}")

def override_get_bot_context():
    # Mock de la fonction pour toujours forcer la réponse
    def always_reply(*args, **kwargs):
        return "Always reply override active."
    return always_reply

def clean_memory():
    gc.collect()
    tracemalloc.reset_peak()

# ====== Routine de benchmark ======

def benchmark():
    prompts = load_prompts(PROMPTS_FILE)
    print(f"📝 {len(prompts)} prompts chargés pour le benchmark.")

    for model_name in MODELS:
        print(f"\n🚀 Benchmarking modèle : {model_name}")
        for prompt_data in prompts:
            user_prompt = prompt_data['prompt']
            metadata = {
                'timestamp': datetime.utcnow().isoformat(),
                'model': model_name,
                'prompt': user_prompt,
            }

            # Override du contexte pour forcer la réponse
            bot_context = override_get_bot_context()

            # Mesurer le temps de réponse
            start_time = time.time()

            try:
                response = ollama.chat(
                    model=model_name,
                    messages=[{'role': 'user', 'content': user_prompt}],
                    options=OLLAMA_OPTIONS
                )
                elapsed_time = time.time() - start_time

                answer = response['message']['content'].strip()

                metadata.update({
                    'response': answer,
                    'elapsed_time_sec': round(elapsed_time, 3),
                    'success': bool(answer),
                })

                # Log et envoi Discord
                print(f"✅ Réponse : {answer[:80]}...")
                send_to_discord(f"🧩 [{model_name}] {answer}")

            except Exception as e:
                metadata.update({
                    'error': str(e),
                    'success': False
                })
                print(f"❌ Erreur pour {model_name}: {e}")
                send_to_discord(f"⚠️ Erreur pour {model_name}: {e}")

            # Sauvegarde du résultat
            save_result(metadata, RESULTS_FILE)

            # Nettoyage mémoire après chaque prompt
            clean_memory()

    print("\n🎉 Benchmark terminé.")

# ====== Exécution principale ======

if __name__ == '__main__':
    benchmark()
