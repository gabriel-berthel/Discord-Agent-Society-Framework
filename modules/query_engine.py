import ollama
import re
from utils.agent_utils import *

OPTIONS = {
    "mirostat": 2,
    "mirostat_tau": 7, 
    "num_predict": 300,
    "mirostat_eta": 0.1, 
    "num_ctx": 4096,
    "repeat_penalty": 1.3,
    "presence_penalty": 1.4,
    "frequency_penalty": 0.2,
    "stop": ["<|endoftext|>"]
}

QUERY_BASE = f"""
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

class QueryEngine():
    """
    QueryEngine class generates natural language queries based on Discord messages and user context.

    This class uses a language model to produce context-aware queries that help retrieve relevant 
    information from a user's notebook or diary, formatted in a natural human-like way.
    
    Context Queries (used in planning refinement) are purely neutral, 
    while response (used to forge response) queries inject agent related information into the prompt.
    """

    def __init__(self, model):
        self.model = model

    async def context_query(self, messages):
        """
        Generates queries based on a list of Discord messages only.

        Args:
            messages (list): A list of message strings from a conversation.

        Returns:
            list: A list of cleaned query strings.
        """
        if messages:
            
            msgs = '\n'.join(messages)

            response = await ollama.AsyncClient().generate(
                model=self.model,
                system=QUERY_BASE,
                prompt=msgs,
                options=OPTIONS
            )
            return self.split_queries(response['response'])

        return []

    async def response_queries(self, plan, context, personality, messages=['No message at the moment.']):
        """
        Generates queries using current plan, user context, and personality traits.

        Args:
            plan (str): The current plan or objective.
            context (str): Relevant notebook or diary context.
            personality (str): Description of the assistant's personality.
            messages (list, optional): Discord messages. Defaults to a placeholder.

        Returns:
            list: A list of cleaned query strings.
        """
        msgs = '\n'.join(messages)
        
        system_instruction = f"""
Your personality is as follows:
{personality}
    
Your current plan is:
{plan}

Here is the context from your notebook or diary:
{context}

---

{QUERY_BASE}
"""
        
        response = await ollama.AsyncClient().generate(
            model=self.model,
            system=system_instruction,
            prompt=msgs,
            options=OPTIONS
        )
        
        return self.split_queries(response['response'])
    
    def split_queries(self, txt):
        """
        Splits raw model output into individual, cleaned queries.

        Args:
            txt (str): Raw response text containing multiple queries.

        Returns:
            list: A list of parsed and cleaned queries.
        """
        
        queries = re.findall(r'Query:\s*(.+?)(?=\nQuery:|\Z)', txt, flags=re.DOTALL)
        return [
            re.sub(r'\s+', ' ', re.sub(r'[^\w\s?]', '', q.strip()))
            for q in queries if q.strip()
        ]