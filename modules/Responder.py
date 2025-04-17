import ollama
from utils.utils import *
import re


OPTIONS = {
    "temperature": 0.9,
    "top_p": 0.85,
    "repeat_penalty": 1.2,
    "presence_penalty": 0.4,
    "frequency_penalty": 0.2,
    "num_predict": 256,
    "mirostat": 0,
    "stop": ["\nUser:", "\nAssistant:", "<|end|>", "\n\n"]
}

class Responder:
    def __init__(self, model):
        self.model = model

    async def respond(self, plan, context, memories, messages, argent_base_prompt):
        
        msgs = '\n'.join(messages)
        prompt = f"""
        You a discord user
        {argent_base_prompt}  
          
        {context}
            
        Taking this into consideration: 
        {plan}
        
        And using the following memories:
        {memories}
        
        You are replying to : {msgs} 
        """

        response = await ollama.AsyncClient().generate(
            model=self.model,
            prompt=prompt,
            options=OPTIONS
        )

        return self.clean_response(response['response'])

    async def new_discussion(self, plan, argent_base_prompt):
        prompt = f"""
        You are a Discord user.
        {argent_base_prompt}
        
        Your plan is:
        {plan}

        Please spark a new topic for discussion immediately.
        """

        response = await ollama.AsyncClient().generate(
            model=self.model,
            prompt=prompt,
            options=OPTIONS
        )

        return self.clean_response(response['response'])

    def clean_response(self, response):
        cleaned_text = response.replace('[', '').replace(']', '').strip('"').strip("'").replace('\n', ' ').strip()
        cleaned_text = re.sub(r"(?i)^(\*{0,2}name\*{0,2}:)\s*", "", cleaned_text)
        return cleaned_text