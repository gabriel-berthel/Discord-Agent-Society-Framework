import json
from collections import defaultdict

def load_results(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_results(baseline_results, agent_results):
    summary = defaultdict(lambda: defaultdict(list))

    for result in baseline_results:
        task = result['task']
        summary['baseline'][task].append(result)

    for result in agent_results:
        task = result['task']
        archetype = result['archetype']
        summary[archetype][task].append(result)

    print("\n📊 Résumé de l'analyse :")
    for archetype, tasks in summary.items():
        print(f"\n🔍 Archetype: {archetype}")
        for task, results in tasks.items():
            avg_time = sum(r['response_time'] for r in results) / len(results)
            avg_length = sum(r['response_length'] for r in results) / len(results)
            print(f"  Task: {task}")
            print(f"    - Nombre de réponses: {len(results)}")
            print(f"    - Temps de réponse moyen: {avg_time:.2f}s")
            print(f"    - Longueur moyenne des réponses: {avg_length:.2f} mots")

if __name__ == '__main__':
    baseline_file = input("Chemin vers le fichier de résultats baseline (.json) : ").strip()
    agents_file = input("Chemin vers le fichier de résultats agents (.json) : ").strip()

    baseline_results = load_results(baseline_file)
    agent_results = load_results(agents_file)

    analyze_results(baseline_results, agent_results)
