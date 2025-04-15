import yaml
import sys


def load_yaml(file_path):
    """Loads a YAML file and returns its contents as a dictionary."""

    try:
        with open(file_path, "r") as file:
            return yaml.safe_load(file)
    except yaml.YAMLError as e:
        print(f"Error loading YAML file '{file_path}': {e}", file=sys.stderr)
        sys.exit(1)

def format_llm_prompt(role, content):
    return {'role': role, 'content': content}

def format_llm_prompts(messages):
    return [format_llm_prompt(role, content) for role, content in messages]

def list_to_text(lst):

    if len(lst) > 0:
        return "- " + lst[0] + "\n- ".join(lst[1:])
    else:
        return "- This section is empty."
