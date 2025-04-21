import ollama
from utils.utils import *
import re

OPTIONS = {
    "mirostat": 2,
    "mirostat_tau": 10, 
    "num_predict": 70,
    "mirostat_eta": 0.1, 
    "num_ctx": 8192,
    "repeat_penalty": 1.5,
    "presence_penalty": 1.5,
    "frequency_penalty": 0.2,
    "penalize_newline": False,
    "stop": ["\n"]
}

class Responder:
    def __init__(self, model):
        self.model = model

    async def respond(self, plan, context, memories, messages, agent_base_prompt, last_messages=[]):
        
        msgs = '\n'.join(messages)

        if last_messages:
            last_msgs = '\n'.join(last_messages)
        else:
           last_msgs = "No previous message."
        
        if memories:
            memories = '\n'.join(memories)
        else:
            memories = "No memories"
        
        try:
            system_instruction = f"""
You are a Discord user with the following personality:
{agent_base_prompt}

What you were planning on doing:
{plan} 

What you can remember:
{memories}

----

The last 5 messages your sent were:
{last_msgs}

{context}

Skip the greetings. You're reading the chat and responding as you feel. 
Reply immediately but don't repeat yourself or what is being said. 
Bring new beef to the table! Keep responses brief, like 1–2 sentences max, like a Discord message, unless maybe a longer answer is really needed.
"""

            response = await ollama.AsyncClient().generate(
                model=self.model,
                system=system_instruction,
                prompt = f"\n{msgs}",
                options=OPTIONS,
                stream=False
            )
        
        except Exception as e:
            print(e)
            
        return self.clean_response(response['response'])

    async def new_discussion(self, plan, argent_base_prompt):
        
        system_instruction = f"""
        {argent_base_prompt}
        """
        
        prompt = f"""
        No one is talking so maybe you should start a new discussion! Just be spontanous and tell us about what u like or want to do or were doing!
        """
        
        response = await ollama.AsyncClient().generate(
            model=self.model,
            prompt=prompt,
            system=system_instruction,
            options=OPTIONS
        )
        
        return self.clean_response(response['response'])


    def clean_response(self, response):
        
        response = re.sub(r'\s+', ' ', response).strip()
        
        if response.startswith("**"):
            cleaned_text = re.sub(r"^\*\*.*?\*\*", "", response)
        elif response.startswith("*"):
            cleaned_text = re.sub(r"^\*.*?\*", "", response)
        elif response.startswith("["):
            cleaned_text = re.sub(r"^\[.*?\]:?\s", "", response)
        elif response.startswith("("):
            cleaned_text = re.sub(r"^\((.*?)\):?\s", "", response)
        elif ":" in response[:25]:
            cleaned_text = re.sub(r"^[^:]+:", "", response)
        else:
            cleaned_text = response
        
        cleaned_text = re.sub(r'(?:#\w+|:\w+:)', '', cleaned_text)
        #  cleaned_text = re.sub(r'[^\w\s,.\-!?]', '', cleaned_text) # remove emojis
        cleaned_text = cleaned_text.strip().replace('\n', ' ').strip().removeprefix('"').removesuffix('"').removeprefix('"').removesuffix('"').removeprefix("[").removesuffix("]").removeprefix(':')
   
        return cleaned_text.strip()