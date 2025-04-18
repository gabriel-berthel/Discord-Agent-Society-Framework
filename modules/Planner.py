import ollama
from utils.utils import *

planner_base = """
Alright, imagine you’re jotting down some thoughts in your personal notebook, but you’re being real with yourself. 
No fluff—just straight-up reflection on where you’re at and what you want going forward.

Start with statements like: “I want to…”, “I’d like to try…”, or “I’m curious to see if…”
These should show not only what you want to do, but also why it matters to you right now.

Reflect on:
- Past decisions, plans, or moments that are shaping your direction
- Lessons learned that have shifted your perspective
- How recent choices align (or don’t) with your bigger goals
- Emotions, doubts, or motivations that are pushing you forward

Be honest and raw—this is about clarifying your direction and capturing your current mindset.
Base your entry on your memories, context, and prior plans.
"""

OPTIONS = {
    "mirostat": 2,
    "mirostat_tau": 8, 
    "mirostat_eta": 0.1, 
    "num_ctx": 8000,
    "repeat_penalty": 1.5,
    "presence_penalty": 1.5,
    "stop": ["."],
    "num_predict": 500
}

class Planner:
    def __init__(self, model):
        self.model = model

    async def refine_plan(self, plan, context, memories, channel_context, argent_base_prompt):
        system_instruction = f"""
        {argent_base_prompt}
        
        Channel context:
        {channel_context}
        
        My previous plan:
        {plan}
        
        Memories:
        {memories}

        Current context:
        {context}
        """
        
        memories = '\n'.join(memories)
        
        prompt = f"""{planner_base}"""

        response = await ollama.AsyncClient().generate(
            model=self.model,
            prompt=prompt,
            system=system_instruction,
            options=OPTIONS
        )

        return response['response']
