import ollama
import re
from utils import *

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
            prompts = [
                ('system', base),
                ('system', 'Here is the discord conversation to write query about:'),
            ] + [('user', msg) for msg in messages]

            response = await ollama.AsyncClient().chat(
                model=self.model,
                messages=format_llm_prompts(prompts)
            )
            
            return self.split_queries(response['message']['content'])
        
        return []

    async def response_queries(self, plan, context, messages=['No message at the moment.']):
        prompts = [
            ('assistant', f'{plan}'),
            ('assistant', f'{context}'),
            ('system', base),
            ('system', f'Now write queries for the following messages:'),
        ] + [('user', msg) for msg in messages]

        response = await ollama.AsyncClient().chat(
            model=self.model,
            messages=format_llm_prompts(prompts)
        )
            
        return self.split_queries(response['message']['content'])

    async def web_queries(self, plan, context, messages=['No message at the moment.']):
        prompts = [
            ('system', f'Your current plan is:\n{plan}'),
            ('system', f'From the conversation, you made the following observations:\n{context}'),
            ('system', web_base),
            ('system', f'Now write queries for the following messages:'),
        ] + [('user', msg) for msg in messages]

        response = await ollama.AsyncClient().chat(
            model=self.model,
            messages=format_llm_prompts(prompts)
        )
            
        return self.split_queries(response['message']['content'])

    def split_queries(self, txt):
        queries = re.findall(r'Query:\s*(.*?)\s*(?=\n|$)', txt)
        cleaned_queries = [
            re.sub(r'\s+', ' ', re.sub(r'[^\w\s?]', '', query.strip())).strip()
            for query in queries if query.strip() != ""
        ]

        return cleaned_queries
