import yaml
import sys
from types import SimpleNamespace
import re

class DictToAttribute(SimpleNamespace):
    """SimpleNameSpace + get method compability :D"""
    def get(self, key, default=None):
        return getattr(self, key, default)

def load_yaml(file_path):
    """Loads a YAML file and returns its contents as a dictionary."""

    try:
        with open(file_path, "r") as file:
            return yaml.safe_load(file)
    except yaml.YAMLError as e:
        print(f"Error loading YAML file '{file_path}': {e}", file=sys.stderr)
        sys.exit(1)
        
def clean_module_output(text: str) -> str:
    """
    Cleans and formats the input text to remove unnecessary whitespace and ensure proper punctuation spacing.

    This function performs the following operations:
    1. Replaces all newlines (`\n`) with a space.
    2. Collapses multiple consecutive spaces into a single space.
    3. Ensures that punctuation marks (.,!?;:) are not preceded by spaces.

    Args:
        text (str): The input text that needs to be cleaned.

    Returns:
        str: The cleaned and formatted text.
    """
    
    text = text.replace('\n', ' ')
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s+([.,!?;:])', r'\1', text)
    return text.strip()
