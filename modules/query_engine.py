import ollama

from configs.ollama_options import QUERIES_OPTIONS
from utils.agent.agent_utils import split_queries, _wait_time_out
from utils.agent.base_prompts import query_prompt_base


class QueryEngine:
    """
    QueryEngine class generates natural language queries based on Discord messages and user context.

    This class uses a language model to produce context-aware queries that help retrieve relevant 
    information from a user's notebook or diary, formatted in a natural human-like way.
    
    Context Queries (used in planning refinement) are purely neutral, 
    while response (used to forge response) queries inject agent related information into the prompt.
    """

    def __init__(self, model):
        self.model = model

    async def create_transcript_queries(self, messages):
        """
        Generates queries based on a list of Discord messages only.

        Args:
            messages (list): A list of message strings from a conversation.

        Returns:
            list: A list of cleaned query strings.
        """
        if messages:
            msgs = '\n'.join(messages)

            response = await _wait_time_out(
                ollama.AsyncClient().generate(
                    model=self.model,
                    prompt=msgs,
                    system=query_prompt_base,
                    options=QUERIES_OPTIONS
                ),
                timeout=120,
                timeout_message="Query Generation Aborted!",
                default_return=[]
            )

            return split_queries(response['response'])

        return []

    async def create_response_queries(self, plan, context, personality, messages=None):
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
        if messages is None:
            messages = ['No message at the moment.']

        msgs = '\n'.join(messages)

        system_instruction = f"""
Your personality is as follows:
{personality}
    
Your current plan is:
{plan}

Here is the context from your notebook or diary:
{context}

---

{query_prompt_base}
"""

        response = await _wait_time_out(
            ollama.AsyncClient().generate(
                model=self.model,
                prompt=msgs,
                system=system_instruction,
                options=QUERIES_OPTIONS
            ),
            timeout=120,
            timeout_message="Query Generation Aborted!",
            default_return=[]
        )

        return split_queries(response['response'])
