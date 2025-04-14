import json
import matplotlib.pyplot as plt
from collections import defaultdict
from fuzzywuzzy import fuzz

def load_results(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return json.load(f)

def compute_accuracy(results):
    metrics = defaultdict(lambda: {"correct": 0, "total": 0})
    for entry in results:
        if str(entry["predicted_label"]) == str(entry["expected_answer"]):
            metrics[entry["archetype"]]["correct"] += 1
        metrics[entry["archetype"]]["total"] += 1
    return {arch: data["correct"] / data["total"] if data["total"] > 0 else 0 for arch, data in metrics.items()}

def compute_fuzzy(results):
    metrics = defaultdict(list)
    for entry in results:
        score = fuzz.ratio(str(entry["response"]).lower(), str(entry["expected_answer"]).lower())
        metrics[entry["archetype"]].append(score)
    return {arch: sum(scores) / len(scores) if scores else 0 for arch, scores in metrics.items()}

def plot_metrics(metrics, title, ylabel):
    plt.figure()
    plt.bar(metrics.keys(), metrics.values())
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("Archétype")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

def main():
    filename = input("Nom du fichier JSON compilé : ")
    results = load_results(filename)

    accuracy = compute_accuracy(results)
    fuzzy = compute_fuzzy(results)

    print("Accuracy:", accuracy)
    print("Fuzzy Matching:", fuzzy)

    plot_metrics(accuracy, "Accuracy par archétype", "Accuracy")
    plot_metrics(fuzzy, "Fuzzy Matching Score par archétype", "Fuzzy Score")

if __name__ == '__main__':
    main()
