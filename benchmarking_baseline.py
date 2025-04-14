# import asyncio
# import json
# import time
# from datetime import datetime
# import promptbench as pb
# import ollama

# MODEL = "llama3"

# # ✅ Fonction pour interroger Ollama
# async def query_ollama(prompt):
#     start_time = time.time()
#     response = ollama.chat(model=MODEL, messages=[{"role": "user", "content": prompt}])
#     end_time = time.time()
#     return response['message']['content'].strip(), end_time - start_time

# # ✅ Cache pour éviter de redétecter les clés à chaque sample
# key_cache = {}

# def get_input_output(dataset_name, sample):
#     if dataset_name in key_cache:
#         input_key, output_key = key_cache[dataset_name]
#     else:
#         possible_input_keys = ["input", "text", "sentence", "premise", "question", "context", "content"]
#         possible_output_keys = ["output", "label", "answer"]

#         input_key = next((key for key in possible_input_keys if key in sample), None)
#         output_key = next((key for key in possible_output_keys if key in sample), None)

#         if input_key is None or output_key is None:
#             print(f"❌ Clés non détectées pour dataset '{dataset_name}'. Sample keys: {list(sample.keys())}")
#             print(f"Sample complet : {sample}")
#             raise ValueError(f"Impossible de détecter les clés input/output pour le dataset '{dataset_name}'.")

#         print(f"✅ Clés détectées pour dataset '{dataset_name}': input = '{input_key}', output = '{output_key}'")
#         key_cache[dataset_name] = (input_key, output_key)

#     return sample[input_key], sample[output_key]

# # ✅ Fonction principale de benchmark baseline
# async def run_baseline():
#     results = []

#     # Liste des datasets PromptBench à utiliser
#     datasets = ["sst2", "bool_logic"] 

#     for dataset_name in datasets:
#         print(f"\n🚀 Benchmarking dataset: {dataset_name}")

#         dataset = pb.DatasetLoader.load_dataset(dataset_name)

#         for sample in dataset:
#             try:
#                 prompt, expected_answer = get_input_output(dataset_name, sample)
#             except ValueError as e:
#                 print(e)
#                 continue  # Passe au sample suivant si non compatible

#             response, response_time = await query_ollama(prompt)

#             results.append({
#                 "task": dataset_name,
#                 "archetype": "baseline",
#                 "input": prompt,
#                 "response": response,
#                 "expected_answer": expected_answer,
#                 "response_time": response_time,
#                 "response_length": len(response.split())
#             })

#     filename = f"baseline_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
#     with open(filename, 'w', encoding='utf-8') as f:
#         json.dump(results, f, ensure_ascii=False, indent=2)

#     print(f"\n✅ Baseline benchmark terminé. Résultats sauvegardés dans '{filename}'.")

# if __name__ == '__main__':
#     asyncio.run(run_baseline())
import promptbench as pb
import json
from datetime import datetime
from tqdm import tqdm
import ollama

RESULTS = []

def prompt_ollama(prompt):
    
    return ollama.generate("llama3.2", prompt)["response"]



def get_projection_fn():
    return lambda pred: 1 if "positive" in pred.lower() else 0 if "negative" in pred.lower() else -1

def run_baseline():
    datasets = ["sst2"]

    for dataset_name in datasets:
        dataset = pb.DatasetLoader.load_dataset(dataset_name)
        prompts = pb.Prompt([f"Classify the sentence as positive or negative: {{content}}"])
        projection_fn = get_projection_fn()

        for prompt in prompts:
            for data in tqdm(dataset, desc=f"Baseline - {dataset_name}"):
                input_text = pb.InputProcess.basic_format(prompt, data)
                label = data['label']
                raw_pred = prompt_ollama(input_text)
                pred = pb.OutputProcess.cls(raw_pred, projection_fn)

                RESULTS.append({
                    "task": dataset_name,
                    "archetype": "baseline",
                    "input": input_text,
                    "response": raw_pred,
                    "predicted_label": pred,
                    "expected_answer": label,
                })

    filename = f"baseline_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(RESULTS, f, ensure_ascii=False, indent=2)
    print(f"✅ Résultats sauvegardés dans '{filename}'.")

if __name__ == '__main__':
    run_baseline()
