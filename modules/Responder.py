import ollama
from utils.utils import *

class Responder():
    def __init__(self, model):
        self.model = model
    
    async def respond(self, plan, context, memories, messages, argent_base_prompt):

        prompts = [
            ('system', argent_base_prompt),
            ('system', f'Your plan: {plan}'),
            ('system', f'Some context: {context}'),
            ('system', f'Your memories: {memories}'),
            ('system', 'Please reply immediately with the content of your response. Do not add any labels or prefixes.')
        ]
        prompts += [('user', msg) for msg in messages]

        response = await ollama.AsyncClient().chat(
            model=self.model, 
            messages=format_llm_prompts(prompts)
        )
        
        return response['message']['content']
    
    async def new_discussion(self, plan, argent_base_prompt):

        prompts = [
            ('system', argent_base_prompt),
            ('system', f'Your plan: {plan}'),
            ('system', f'Please spark a new topic immediately. Do not add any labels or prefixes.'),
        ]

        response = await ollama.AsyncClient().chat(
            model=self.model, 
            messages=format_llm_prompts(prompts)
        )

        return response['message']['content']
