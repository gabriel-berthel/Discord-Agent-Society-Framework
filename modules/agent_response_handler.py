import ollama

from configs.ollama_options import AGENT_RESPONSE_OPTIONS
from utils.agent.agent_utils import clean_response


class Responder:
    """
    Responder class provides functionality for generating discord-like responses.
    Responses are made based on context, plan, memories, and messages.

    Methods:
    - respond: Generates a response based on context, plan, memories, and recent messages.
    - new_discussion: Initiates a new discussion with a spontaneous subject.
    - clean_response: Cleans and formats the generated response.
    """

    def __init__(self, model):
        self.model = model

    async def respond(self, plan, context, memories, messages, agent_base_prompt, last_messages=None):
        """
        Generates a response to a Discord conversation based on the provided context, plan, memories, and messages.

        Args:
            plan (str): The current plan or intention for the conversation.
            context (str): The relevant context from the ongoing conversation.
            memories (list): A list of relevant memories or previous interactions.
            messages (list): A list of the latest messages to be included in the response.
            agent_base_prompt (str): The personality description of the responder.
            last_messages (list, optional): A list of the last 5 messages sent by the user.

        Returns:
            str: A concise, context-aware response.
        """

        if last_messages is None:
            last_messages = []

        msgs = '\n'.join(messages)

        if last_messages:
            last_msgs = '\n'.join(last_messages)
        else:
            last_msgs = "No previous message."

        if memories:
            memories = '\n'.join(memories)
        else:
            memories = "No memories"

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
            prompt=f"\n{msgs}",
            options=AGENT_RESPONSE_OPTIONS,
            stream=False
        )

        return clean_response(response['response'])

    async def new_discussion(self, plan, argent_base_prompt):
        """
        Initiates a new discussion barely inputing anything to the model so it's very "random".

        Args:
            plan (str): The current plan or intention for the conversation.
            argent_base_prompt (str): The personality description of the responder.

        Returns:
            str: A spontaneous message to start a new discussion.
        """

        system_instruction = f"""
        {argent_base_prompt}
        
        You plan was to {plan}
        """

        prompt = f"""
        No one is talking so maybe you should start a new discussion! Just be spontanous and tell us about what u like or want to do or were doing!
        """

        response = await ollama.AsyncClient().generate(
            model=self.model,
            prompt=prompt,
            system=system_instruction,
            options=AGENT_RESPONSE_OPTIONS
        )

        return clean_response(response['response'])
