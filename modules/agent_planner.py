import ollama

from configs.ollama_options import AGENT_PLANNING_OPTIONS
from utils.agent.agent_utils import *
from utils.agent.base_prompts import planner_base


class Planner:
    """
    Planner class is used to generate reflections and refine plans based on context, memories, and previous decisions.

    Methods:
    - refine_plan: Refines and generates a new plan based on the provided context, memories, channel context, and prior plans.
    """

    def __init__(self, model):
        self.model = model

    async def make_plan(self, plan, context, memories, channel_context, argent_base_prompt):
        """
        Refines and generates a new plan based on current context, memories, previous decisions, and channel context.

        Args:
            plan (str): The initial plan or goal to be refined.
            context (str): The current context relevant to the plan.
            memories (list): A list of relevant memories influencing the current plan.
            channel_context (str): Specific context related to the communication channel.
            argent_base_prompt (str): The personality description or agent base prompt.

        Returns:
            str: A refined plan based on reflection and context.
        """

        memories = '\n'.join(memories)

        system_instruction = f"""
        {argent_base_prompt}
        
        Channel context:
        {channel_context}
        
        My previous plan:
        {plan}
        
        Memories:
        {memories}
        """

        prompt = f"""
        {planner_base}
        
        Current context:
        {context}
        """

        response = await ollama.AsyncClient().generate(
            model=self.model,
            prompt=prompt,
            system=system_instruction,
            options=AGENT_PLANNING_OPTIONS
        )

        return clean_module_output(response['response'])
