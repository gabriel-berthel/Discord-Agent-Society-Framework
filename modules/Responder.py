import ollama
from utils.utils import *
import re


OPTIONS = {
    "temperature": 0.8,
    "top_p": 0.9,
    "repeat_penalty": 1.1,
    "presence_penalty": 0.6,
    "frequency_penalty": 0.3,
    "num_predict": 200,
    "mirostat": 0,
    "stop": ["\nUser:", "\nAssistant:", "<|end|>", "\n\n"]
}

class Responder:
    def __init__(self, model):
        self.model = model

    async def respond(self, plan, context, memories, messages, argent_base_prompt):
        
        
        msgs = '\n'.join(messages)
        # memories = '\n'.join(memories)
        
        if memories:
            memories = '\n'.join(memories)
        else:
            memories = "No memories"
        
        try:
            system_instruction = f"""
            You are a Discord user with the following personality:
            {argent_base_prompt}

            Here’s what’s going on in the conversation:
            {context}

            Here's what you planned to do next:
            {plan} 

            Here are relevant memories that may help:
            {memories}
            """
            
            response = await ollama.AsyncClient().generate(
                model=self.model,
                system=system_instruction,
                prompt=f"Reply immediately to:\n{msgs}",
                options=OPTIONS
            )

        except Exception as e:
            print(e)
            
        return self.clean_response(response['response'])

    async def new_discussion(self, plan, argent_base_prompt):
        
        
        system_instruction = f"""
        You are a Discord user with the following personality:
        {argent_base_prompt}

        Taking this into consideration, here's what you planned to do next:
        {plan} 
        """
        
        prompt = f"""
        Spark a new discussion as a spontanous message.
        """
        
        response = await ollama.AsyncClient().generate(
            model=self.model,
            prompt=prompt,
            system=system_instruction,
            options=OPTIONS
        )
        
        return self.clean_response(response['response'])


    def clean_response(self, response):
    
        # Removes prefix
        if response.startswith("**"):   
            cleaned_text = re.sub(r"^\*\*(.+?)\*\*", "", response)
        else:
            cleaned_text = re.sub(r"^(.*?):\s", "", response)
            
        cleaned_text = cleaned_text.strip()
        cleaned_text = re.sub(r'^"(.*)"$', r'\1', cleaned_text)


        return cleaned_text