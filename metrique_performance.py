import json
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
from rapidfuzz import fuzz

def load_results(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def compute_accuracy(results):
    correct = 0
    total = 0
    for r in results:
        if 'expected_answer' in r:
            total += 1
            if r['expected_answer'].strip().lower() == r['response'].strip().lower():
                correct += 1
    return (correct / total) * 100 if total > 0 else 0

def compute_fuzzy_score(results):
    scores = []
    for r in results:
        if 'expected_answer' in r:
            expected = r['expected_answer'].strip().lower()
            response = r['response'].strip().lower()
            score = fuzz.ratio(expected, response)
            scores.append(score)
    return sum(scores) / len(scores) if scores else 0

def analyze_results(baseline_results, agent_results):
    summary = defaultdict(lambda: defaultdict(list))

    # Collect baseline
    for result in baseline_results:
        task = result['task']
        result['source'] = 'baseline'
        summary['baseline'][task].append(result)

    # Collect agents
    for result in agent_results:
        task = result['task']
        archetype = result['archetype']
        result['source'] = archetype
        summary[archetype][task].append(result)

    export_data = []
    print("\n Résumé des performances :")
    for archetype, tasks in summary.items():
        print(f"\n Archetype: {archetype}")
        for task, results in tasks.items():
            num_responses = len(results)
            avg_time = sum(r['response_time'] for r in results) / num_responses
            avg_length = sum(r['response_length'] for r in results) / num_responses
            accuracy = compute_accuracy(results)
            fuzzy_score = compute_fuzzy_score(results)

            print(f"   Task: {task}")
            print(f"    - Nombre de réponses: {num_responses}")
            print(f"    - Temps de réponse moyen: {avg_time:.2f} s")
            print(f"    - Longueur moyenne des réponses: {avg_length:.2f} mots")
            print(f"    - Accuracy: {accuracy:.2f}%")
            print(f"    - Fuzzy ratio moyen: {fuzzy_score:.2f}%")

            export_data.append({
                "archetype": archetype,
                "task": task,
                "num_responses": num_responses,
                "avg_response_time": avg_time,
                "avg_response_length": avg_length,
                "accuracy": accuracy,
                "fuzzy_ratio": fuzzy_score
            })

    # Export CSV
    df = pd.DataFrame(export_data)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_filename = f"benchmark_summary_{timestamp}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"\n Résumé exporté dans '{csv_filename}'.")

    # Plot accuracy
    plt.figure(figsize=(10, 6))
    for archetype in df['archetype'].unique():
        data = df[df['archetype'] == archetype]
        plt.plot(data['task'], data['accuracy'], marker='o', label=archetype)
    plt.title('Accuracy par tâche et par archetype')
    plt.xlabel('Tâche')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    accuracy_plot = f'accuracy_plot_{timestamp}.png'
    plt.savefig(accuracy_plot)
    print(f" Courbe d\'accuracy sauvegardée dans '{accuracy_plot}'.")

    # Plot response time
    plt.figure(figsize=(10, 6))
    for archetype in df['archetype'].unique():
        data = df[df['archetype'] == archetype]
        plt.plot(data['task'], data['avg_response_time'], marker='o', label=archetype)
    plt.title('Temps de réponse moyen par tâche et par archetype')
    plt.xlabel('Tâche')
    plt.ylabel('Temps de réponse moyen (s)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    response_time_plot = f'response_time_plot_{timestamp}.png'
    plt.savefig(response_time_plot)
    print(f"Courbe de temps de réponse sauvegardée dans '{response_time_plot}'.")

    plt.show()

if __name__ == '__main__':
    baseline_file = input("Chemin vers le fichier de résultats baseline (.json) : ").strip()
    agents_file = input("Chemin vers le fichier de résultats agents (.json) : ").strip()

    baseline_results = load_results(baseline_file)
    agent_results = load_results(agents_file)

    analyze_results(baseline_results, agent_results)
