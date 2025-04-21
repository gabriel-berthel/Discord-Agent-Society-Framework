import ollama
from utils.agent_utils import *

planner_base = """
Alright, imagine you’re jotting down some thoughts in your personal notebook, but you’re being real with yourself. 
No fluff—just straight-up reflection on where you’re at and what you want going forward.

Start with statements like: “I want to…”, “I’d like to try…”, or “I’m curious to see if…”
These should show not only what you want to do, but also why it matters to you right now.

Take a moment to reflect on the decisions, plans, or key moments from your past that are influencing where you're headed now. What lessons have surfaced that caused you to see things differently or approach situations in a new way? Think about how your recent choices fit into the bigger picture—are they truly aligned with your long-term goals, or do they reveal a shift in priorities? Pay attention to the emotions, doubts, or inner motivations that are driving you forward. What’s fueling your momentum, and what’s holding your attention as you move ahead?

Be honest and raw—this is about clarifying your direction and capturing your current mindset.
Base your entry on your memories, context, and prior plans and write a paragraphe.
"""

OPTIONS = {
    "mirostat": 2,
    "mirostat_tau": 10, 
    "mirostat_eta": 0.1, 
    "num_ctx": 4096,
    "repeat_penalty": 1.5,
    "presence_penalty": 1.5,
    "frequency_penalty": 0.2,
    "num_predict": 300,
    "stop": ["<|endoftext|>"],
    "newline_penalty": True
}

class Planner:
    """
    Planner class is used to generate reflections and refine plans based on context, memories, and previous decisions.

    Methods:
    - refine_plan: Refines and generates a new plan based on the provided context, memories, channel context, and prior plans.
    """

    def __init__(self, model):
        self.model = model

    async def refine_plan(self, plan, context, memories, channel_context, argent_base_prompt):
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
        
        system_instruction = f"""
        {argent_base_prompt}
        
        Channel context:
        {channel_context}
        
        My previous plan:
        {plan}
        
        Memories:
        {memories}
        """
        
        memories = '\n'.join(memories)
        
        prompt = f"""
        {planner_base}
        
        Current context:
        {context}
        """

        response = await ollama.AsyncClient().generate(
            model=self.model,
            prompt=prompt,
            system=system_instruction,
            options=OPTIONS
        )

        return clean_module_output(response['response'])
