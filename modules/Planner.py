import ollama
from utils import *

base = """
Imagine you're reflecting on your plans and objectives in your personal notebook. 
Based on your previous experiences, decisions, and memories, write a paragraph outlining what you would like to achieve moving forward. 
Your plan should be expressed in the first person, as if you're speaking directly to yourself. 
Begin with statements like, 'I want to do this,' 'I would like to try that,' or 'I want to see if... because...' Incorporate any relevant memories or past plans that could influence your current objectives. 
Consider any recent decisions you've made and explain how they align with what you want to accomplish. 
This is your personal reflection, so feel free to express your thoughts freely and honestly."
"""

class Planner():
    def __init__(self, model):
        self.model = model

    async def refine_plan(self, plan, context, memories, context):
        prompts = [
            ('system', get_base_prompt()),
            ('system', f'You recall that your previous plan and objectives were:\n{plan}'),
            ('system', f'From the current conversation, you made the following observations:\n{context}'),
            ('system', f'Allowing you to search your notebook and remember that: \n{list_to_text(memories)}'),
            ('system', f'{context}'),
            ('system', base),
            ('user', 'Please procede to write your new plan in your personal diary'),
        ]

        response = await ollama.AsyncClient().chat(
            model=self.model,
            messages=format_llm_prompts(prompts)
        )

        return response['message']['content']