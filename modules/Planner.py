import ollama
from utils.utils import *

base = """
Imagine you’re writing a forward-looking entry in your personal notebook, grounded in self-reflection and past experience. This is your space to explore your current goals, intentions, and desires based on what you've learned about yourself over time.

Write a thoughtful paragraph in the first person, as if you're speaking directly to yourself. Begin with phrases like:
“I want to…”, “I would like to try…”, or “I’m curious to see if…”
These statements should express not only what you hope to do, but also why these goals matter to you.

As you write, consider including:
    Memories, previous plans, or decisions that are influencing your current mindset
    Lessons learned from recent experiences or turning points
    How recent choices or shifts in direction align with your evolving objectives
    Any emotions, doubts, or motivations you feel as you look ahead

This entry is a personal reflection—honest, unfiltered, and meaningful to you. Use it as a tool to clarify your direction and to capture a snapshot of your mindset, so you can revisit it later and see how your goals and self-understanding have developed over time.
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