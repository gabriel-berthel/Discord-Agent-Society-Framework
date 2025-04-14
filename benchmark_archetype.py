import promptbench as pb
import json
from datetime import datetime
from tqdm import tqdm

ARCHETYPES = ["trouble_maker", "fact_checker", "activist", "moderator", "baseline"]
RESULTS = []

def get_projection_fn():
    return lambda pred: 1 if "positive" in pred.lower() else 0 if "negative" in pred.lower() else -1

def run_agents_benchmark():
    datasets = ["sst2"]

    for agent_name in ARCHETYPES:
        #model = pb.LLMModel(model="google/flan-t5-large", max_new_tokens=10, temperature=0.0001, device='cpu')
        for dataset_name in datasets:
            dataset = pb.DatasetLoader.load_dataset(dataset_name)
            prompts = pb.Prompt([f"Classify the sentence as positive or negative: {{content}}"])
            projection_fn = get_projection_fn()

            for prompt in prompts:
                for data in tqdm(dataset, desc=f"{agent_name} - {dataset_name}"):
                    input_text = pb.InputProcess.basic_format(prompt, data)
                    label = data['label']
                    raw_pred = model(input_text)
                    pred = pb.OutputProcess.cls(raw_pred, projection_fn)

                    RESULTS.append({
                        "task": dataset_name,
                        "archetype": agent_name,
                        "input": input_text,
                        "response": raw_pred,
                        "predicted_label": pred,
                        "expected_answer": label,
                    })

    filename = f"agents_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(RESULTS, f, ensure_ascii=False, indent=2)
    print(f"✅ Résultats sauvegardés dans '{filename}'.")

if __name__ == '__main__':
    run_agents_benchmark()
