import json
from datetime import datetime
import glob

def compile_results():
    result_files = glob.glob("*.json")
    global_results = []

    for file in result_files:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
            global_results.extend(data)

    output_file = f"compiled_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(global_results, f, ensure_ascii=False, indent=2)

    print(f"✅ Compilation terminée. Résultats dans '{output_file}'.")

if __name__ == '__main__':
    compile_results()
