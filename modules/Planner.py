import ollama
from utils.utils import *

base = """
Alright, imagine you’re jotting down some thoughts in your personal notebook, but you’re being real with yourself. 
No fluff, just straight-up reflection on where you’re at and what you want going forward. 

This is where you dig into your current goals, plans, and desires based on what you’ve learned about yourself so far.

Start by saying stuff like:
“I want to…”, “I’d like to try…”, or “I’m curious to see if…”
These statements should show not only what you want to do, but also why it matters to you right now.

Think about:
    Past decisions, plans, or moments that are influencing where you’re headed
    Any lessons you’ve learned from recent stuff that’s shifted how you see things
    How your choices lately align with your bigger goals (or not)
    What emotions, doubts, or motivations are driving you forward

Keep it honest and raw—this is all about getting to the core of what you want and why you want it. 
Use this as a moment to clarify your direction and capture where you’re at right now, so you can look back later and see how much you've grown.
"""

class Planner():
    def __init__(self, model):
        self.model = model

    async def refine_plan(self, plan, context, memories, channel_context, argent_base_prompt):
        prompts = [
            ('assistant', argent_base_prompt),
            ('assistant', f'{plan}'),
            ('assistant', f'{context}'),
            ('assistant', f'{list_to_text(memories)}'),
            ('system', f'{channel_context}'),
            ('system', base),
            ('user', 'I am waiting for your notebook entry! What is it?'),
        ]

        response = await ollama.AsyncClient().chat(
            model=self.model,
            messages=format_llm_prompts(prompts)
        )

        return response['message']['content']