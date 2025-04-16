import ollama
from utils.utils import *

class Responder():
    def __init__(self, model):
        self.model = model
    
    async def respond(self, plan, context, memories, messages, argent_base_prompt):

        prompts = [
            ('system', argent_base_prompt),
            ('system', f'Your plan:'),
            ('assistant', f'{plan}'),
            ('system', f'Some context:'),
            ('assistant', f'{context}'),
            ('system', f'Your memories:'),
            ('assistant', f'{memories}'),
            ('system', f'Reply right away with the actual message. That’s it.')
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
            ('assistant', f'Your currents plans, goals and objectives are:\n{plan}'),
            ('user', f'Anything nice to talk about?')
        ]

        response = await ollama.AsyncClient().chat(
            model=self.model, 
            messages=format_llm_prompts(prompts)
        )

        return response['message']['content']
