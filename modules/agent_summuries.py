import ollama

from configs.ollama_options import CONTEXTUALIZER_NEUTRAL_OPTIONS, REFLECTIONS_OPTIONS
from utils.agent.agent_utils import clean_module_output, _wait_time_out
from utils.agent.base_prompts import neutral_base, engaged_base


class Contextualizer:
    """
    Contextualizer class generates summaries and reflections from Discord conversations.

    It provides two modes:
    - A neutral, objective summary using a student-like tone (so agents are not goldfishes)
    - Reflections, taking agent personality biaises into account (memories)
    """

    def __init__(self, model):
        self.model = model

    async def summurize_transcript(self, messages, bot_context):
        """
        Generates a neutral, factual summary of a Discord conversation.

        Args:
            messages (list): List of message strings from the conversation.
            bot_context (str): Contextual system prompt for the assistant.

        Returns:
            str: A concise summary paragraph written in a neutral tone.
        """
        msgs = '\n'.join([f"{msg}" for msg in messages])

        system = f"""
        {bot_context}
        """

        prompt = f"""
        {neutral_base}
        
        The transcript to write about immediately:
        {msgs}
        """

        if messages:
            response = await _wait_time_out(
                ollama.AsyncClient().generate(
                    model=self.model,
                    prompt=prompt,
                    system=system,
                    options=CONTEXTUALIZER_NEUTRAL_OPTIONS
                ),
                timeout=60 * 2,
                timeout_message="Summury Generation Aborted! Model waited for 2 minutes",
                default_return="Nothing seems to be happening here."
            )

            return clean_module_output(response['response'])

        return "Reading the discord conversation, I can observe that there is no messages at the moment. I should consider sparking a new topic."

    async def summurize_into_memory(self, messages, agent_base_prompt):
        """
        Produces a reflective, personal note based on a Discord conversation.

        Args:
            messages (list): List of message strings from the conversation.
            agent_base_prompt (str): Assistant's personality or base system prompt.

        Returns:
            str: A personal reflection paragraph written in a casual, introspective tone.
        """

        msgs = '\n'.join([f"{msg}" for msg in messages])

        system = f"""
        {agent_base_prompt}
        """

        prompt = f"""
        {engaged_base}
        
        
        Based on your personality, here is transcript to reflection about:
        {msgs}
        """

        response = await _wait_time_out(
            ollama.AsyncClient().generate(
                model=self.model,
                prompt=prompt,
                system=system,
                options=REFLECTIONS_OPTIONS
            ),
            timeout=60 * 3,
            timeout_message="Reflection Generation Aborted! Model waited for 3 minutes",
            default_return=""
        )

        return clean_module_output(response['response'])
