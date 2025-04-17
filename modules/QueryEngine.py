import ollama
import re
from utils.utils import *

QUERY_ENGINE_OPTIONS = {
    "temperature": 1.0,
    "top_p": 0.85,
    "repeat_penalty": 1.1,
    "presence_penalty": 0.6,
    "frequency_penalty": 0.3,
    "num_predict": 512,
    "mirostat": 0,
    "stop": ["\nUser:", "\nAssistant:", "<|end|>", "\n\n"]
}

base = """
Imagine you are a Discord user who can query your personal notebook and diary to help respond to messages. 
It’s important to ask relevant queries that will assist in crafting appropriate responses. 
When you query, make sure to identify important entities (such as names, dates, or topics) and align your responses with the plan or context you are working with. 
You should ask your queries in natural human language like you are browsing the web, and I will provide relevant information from your notebook and diary.

Please format your queries as follows:

Query: Your first query here  
Query: Your second query here  
Query: Your third query here

These queries will help ensure your responses are informed, relevant, and consistent with the context of your conversations. 
Feel free to ask any questions you may have.
"""

web_base = """
Imagine you are a Discord user who can query the web to help you respond to messages.
It’s important to ask relevant queries that will assist in crafting appropriate responses. 
When you query, make sure to align your responses with the plan or context you are working with. 
You should ask your queries in natural human language like you are browsing the web.

Please format your queries as follows:

Query: Your first query here  
Query: Your second query here  
Query: Your third query here

These queries will help ensure your responses are informed, relevant, and consistent with the context of your conversations. 
Feel free to ask any questions you may have.
"""

class QueryEngine():
    def __init__(self, model):
        self.model = model

    async def context_query(self, messages):
        if messages:
            msgs = '\n'.join([f"{msg}" for msg in messages])
            prompt = f"""
            {base}
            Here is the Discord conversation you need to write queries about:
            {msgs}
            """

            response = await ollama.AsyncClient().generate(
                model=self.model,
                prompt=prompt,
                options=QUERY_ENGINE_OPTIONS
            )
            return self.split_queries(response['response'])

        return []

    async def response_queries(self, plan, context, personality, messages=['No message at the moment.']):
        msgs = '\n'.join([f"{msg}" for msg in messages])
        prompt = f"""
        {base}
        Your personality is as follows:
        {personality}
        
        Your current plan is:
        {plan}
        
        Here is the context from your notebook or diary:
        {context}
        
        Now, write queries for the following messages:
        {msgs}
        """

        response = await ollama.AsyncClient().generate(
            model=self.model,
            prompt=prompt,
            options=QUERY_ENGINE_OPTIONS
        )
        return self.split_queries(response['response'])

    async def web_queries(self, plan, context, messages=['No message at the moment.']):
        
        msgs = '\n'.join([f"{msg}" for msg in messages])
        prompt = f"""
        {web_base}
        Your current plan is:
        {plan}
        
        From the conversation, you made the following observations:
        {context}
        
        Now, write queries for the following messages:
        {msgs}
        """

        response = await ollama.AsyncClient().generate(
            model=self.model,
            prompt=prompt,
            options=QUERY_ENGINE_OPTIONS
        )
        return self.split_queries(response['response'])

    def split_queries(self, txt):
        queries = re.findall(r'Query:\s*(.+?)(?=\nQuery:|\Z)', txt, flags=re.DOTALL)
        return [
            re.sub(r'\s+', ' ', re.sub(r'[^\w\s?]', '', q.strip()))
            for q in queries if q.strip()
        ]
