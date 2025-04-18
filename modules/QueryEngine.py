import ollama
import re
from utils.utils import *

OPTIONS = {
    "mirostat": 2,
    "mirostat_tau": 8, 
    "num_predict": 400,
    "mirostat_eta": 0.1, 
    "num_ctx": 4000,
    "repeat_penalty": 1.5,
    "presence_penalty": 1.5,
    "penalize_newline": True,
    "stop": ["\n"]
}


class QueryEngine():
    def __init__(self, model):
        self.model = model

    async def context_query(self, messages):
        if messages:
            
            msgs = '\n'.join(messages)
            
            system_instruction = f"""
Imagine you are a Discord user who can query your personal notebook and diary to help respond to messages. 
It’s important to ask relevant queries that will assist in crafting appropriate responses. 
When you query, make sure to identify important entities (such as names, dates, or topics) and align your responses with the plan or context you are working with. 
You should ask your queries in natural human language like you are browsing the web, and I will provide relevant information from your notebook and diary.

Please format your queries as follows:

Query: Your first query here  
Query: Your second query here  
Query: Your third query here

Here is the Discord conversation you need to write queries about:
            """

            response = await ollama.AsyncClient().generate(
                model=self.model,
                system=system_instruction,
                prompt=msgs,
                options=OPTIONS
            )
            return self.split_queries(response['response'])

        return []

    async def response_queries(self, plan, context, personality, messages=['No message at the moment.']):
        msgs = '\n'.join(messages)
        
        system_instruction = f"""
Your personality is as follows:
{personality}
    
Your current plan is:
{plan}

Here is the context from your notebook or diary:
{context}

Imagine you are a Discord user who can query your personal notebook and diary to help respond to messages. 
It’s important to ask relevant queries that will assist in crafting appropriate responses. 

When you query, make sure to identify important entities (such as names, dates, or topics) and align your responses with the plan or context you are working with. 
You should ask your queries in natural human language like you are browsing the web, and I will provide relevant information from your notebook and diary.

Please format your queries as follows:

Query: Your first query here  
Query: Your second query here  
Query: Your third query here

Make sure to write queries according to your personality, plan and context
        """
        
        response = await ollama.AsyncClient().generate(
            model=self.model,
            system=system_instruction,
            prompt=msgs,
            options=OPTIONS
        )
        
        return self.split_queries(response['response'])
    
    def split_queries(self, txt):
        queries = re.findall(r'Query:\s*(.+?)(?=\nQuery:|\Z)', txt, flags=re.DOTALL)
        return [
            re.sub(r'\s+', ' ', re.sub(r'[^\w\s?]', '', q.strip()))
            for q in queries if q.strip()
        ]