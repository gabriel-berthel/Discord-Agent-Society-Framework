import pickle
from types import SimpleNamespace

def get_log_dict(log_dict_path):
    with open(log_dict_path, "rb") as f:
        logs = pickle.load(f)
    return SimpleNamespace(**logs)

def prepare_qa_bench_dataset():
    roles = [
        "activist",
        "baseline",
        "fact_checker",
        "mediator",
        "trouble_maker"
    ]
    
    logs_data = {
        role: get_log_dict(f"qa_bench/qa_bench_{role}.pkl") for role in roles
    }

    logs = SimpleNamespace()

    for archetype, logs_object in logs_data.items():
        role_data = SimpleNamespace(
            plans=[(plan['input'], plan['output']) for plan in logs_object.plans],
            reflections=[(reflection['input'], reflection['output']) for reflection in logs_object.reflections],
            context_queries=[(query['input'], query['output']) for query in logs_object.context_queries],
            neutral_ctxs=[(ctx['input'], ctx['output']) for ctx in logs_object.neutral_ctxs],
            response_queries=[(query['input'], query['output']) for query in logs_object.response_queries],
            memories=[(memory['input'], memory['output']) for memory in logs_object.memories],
            summuries=[(summary['input'], summary['output']) for summary in logs_object.summuries],
            web_queries=[(query['input'], query['output']) for query in logs_object.web_queries]
        )
        
        setattr(logs, archetype, role_data)
    
    return logs

logs = prepare_qa_bench_dataset()

print(logs.activist.plans)
print(logs.mediator.reflections)
