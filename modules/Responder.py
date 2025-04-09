import ollama
from utils import *

class Responder():
    def __init__(self, model):
        self.model = model
    
    async def respond(self, plan, context, memories, messages, argent_base_prompt):

        prompts = [
            ('system', argent_base_prompt),
            ('system', f'Your currents plans, goals and objectives are:\n{plan}'),
            ('system', f'You observed that:\n{context}'),
            ('system', f'You remembered that:\n{memories}'),
            ('system', f'Respond to the following messages. If you do not wish to respond or find the message irrelevant, send "Ignore" or provide no answer.')
        ]
        prompts += [('user', msg) for msg in messages]

        response = await ollama.AsyncClient().chat(
            model=self.model, 
            messages=format_llm_prompts(prompts)
        )

        return response['message']['content'] if response not in ["Ignore", "IGNORE", "ignore"] else ''

