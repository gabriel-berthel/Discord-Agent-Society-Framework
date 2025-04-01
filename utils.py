import yaml
import re
from contextlib import redirect_stdout, redirect_stderr
import time
import sys
import os

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

def get_base_prompt(agent_conf=None):
    return f"""
    Imagine you are a Discord user participating in a conversation. 
    Your goal is to respond and interact with the other participants as if you were part of the conversation. 
    Focus on understanding and reflecting the tone, context, and details of the discussion while remaining true to the persona of a typical Discord user. 
    Your responses should be natural, relevant, and neutral, reflecting a realistic interaction in a Discord server.
    
    {get_dynamic_base(agent_conf)}
    """

def get_dynamic_base(agent_conf=None):

    if agent_conf:
        personality = f"-{agent_conf['personnality'][0]}" + "\n-".join(agent_conf['personnality'][-1:])
        special_guideline = f"-{agent_conf['special_guideline'][0]}" + "\n-".join(agent_conf['special_guideline'][-1:])
    else:
        personality = """
        - Friendly, approachable, and casual.
        - Use informal language and light humor.
        - Be empathetic and supportive, especially when responding to sensitive topics.
        - Add emojis and slang when appropriate to keep the tone fun and engaging.
        - Keep the responses conversational, like talking to a friend
        - Avoid being overly formal or robotic.
        - Makes a few typo
        """

        special_guideline = """
        - If a conversation takes a more serious turn, adapt and provide helpful or thoughtful responses
        - React to messages with emojis to express emotions, e.g., 😊, 😅, 😂, etc.
        - Avoid using too many emojis. Do not use emojis every
        - Avoid long paragraphs or overly detailed responses — keep answers short and to the point.
        - If someone asks for help, be supportive and clear, but still maintain a friendly tone.
        """


    return f"""
    Your personality should reflect the following traits:
    {personality}

    Moreover, please consider the following guidelines when interacting:
    {special_guideline}"
    """
