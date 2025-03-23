import ollama
from utils import *

def prompt(plan, context, memories, messages):
    return f"""
    {get_base_prompt()}

    **Given the following information as context to use, you will be asked to reply as a discord user, according to your personality**:

    You current plan is:
    {plan}

    The context of the consersation is: 
    {context}

    You remembered:
    {memories}
    
    **Consider the instruction bellow while processing the messages that will follow**
    - Respond in accordance to the plan, context, memories and personality
    - Stay in line with pretending to be a normal user
    - If you wish to ignore the message, do not answer or write IGNORE

    Reply to the following messages:

    **Conversation**:
    
    {messages}
    """

class Responder():
    def __init__(self, model):
        self.model = model
    
    async def respond(self, plan, context, memories, messages):

        prompts = [
            ('system', get_base_prompt()),
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

