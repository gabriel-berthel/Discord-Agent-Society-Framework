import ollama
import re
from utils import *

def prompt(content, guidelines, context):
    return f"""
    Given the context below, your task is to create a list of **human-friendly** and **relevant** natural language queries. These queries should help retrieve useful information based on the context history and could be helpful for future reference.

    You must act in evaluate relevancy in accordance to your personality, goals and values.
    {get_dynamic_base()}
    
    **Guidelines for Creating Queries:**
    - Create **short, concise, and natural language queries** that make sense in the context of the context.
    - Focus on extracting the topic and information that may be relevant later.
    - Each query should be **specific** to the conversation and avoid being too broad or vague.
    - The queries should be in the form of a **question or statement**, as they would be naturally asked by a person on a search engine.
    - Ensure queries are **easy to understand** and **written in a simple, conversational tone**.
    - Do **not** include system messages, internal thoughts, placeholders, or any unnecessary metadata.
    - Do **not** provide explanations, system thoughts, or any extra information.
    - Keep the queries **short and clear**—avoid adding extra commentary or information.
    - Each query should be **one line**, written in **simple human-readable text**, with no backticks, quotation marks, or special formatting.
    - Do not include empty queries or irrelevant information.
    - **Name the users** and **name entities**

    **While processing the context you will take into account the following information:**
    {guidelines}

    Here is the {context} to reference:

    {content}

    From the above {context}, please output between 1 and 10 relevant queries in the following format:

    Query: Your first query here
    Query: Your second query here
    Query: Your third query here
    """

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
Feel free to ask any questions you may have, and I'll provide answers based on the entries in your notebook and diary.
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
            ('system', f'Your current plan is:\n{plan}'),
            ('system', f'From the conversation, you made the following observations:\n{context}'),
            ('system', base),
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
