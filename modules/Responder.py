import ollama
from utils import *

class Responder():
    def __init__(self, model):
        self.model = model
    
    async def respond(self, plan, context, memories, messages, argent_base_prompt):

        prompts = [
            ('system', argent_base_prompt),
            ('assistant', f'{plan}'),
            ('assistant', f'{context}'),
            ('assistant', f'{memories}'),
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
            ('system', f'Your currents plans, goals and objectives are:\n{plan}'),
            ('system', f'You decided to randomly spark a new discussion, sharing personal thoughts and reflections. Let your thoughts unfold freely—this is a moment to explore whatever feels present, meaningful, or unresolved. Speak in the first person, as if writing in your personal notebook or beginning a thoughtful conversation with yourself.')
        ]

        response = await ollama.AsyncClient().chat(
            model=self.model, 
            messages=format_llm_prompts(prompts)
        )


        return response['message']['content']
