import ollama
from utils.utils import *

neutral_base = """
You are a student summarizing a Discord conversation. 
Your goal is to create a clear and neutral summary, using first-person language for your contributions (when your name appear).

Focus on:
- Key points and decisions made in the conversation.
- Names of people, companies, events, or any identifiable entities.
- Keep the tone objective and factual—avoid opinions or analysis.

Start with:
"Reading the Discord conversation, I can observe that..."

Keep the summary concise and fact-based, ideally around 1024 characters.
"""

engaged_base = """
Imagine you’re jotting down a quick reflective note after a chat, like you’re talking to yourself on Discord. This is where you get to think about how the convo shifted your mindset or changed your vibe.

Here’s what to think about:

- What did we talk about, and what hit you the most?  
- Any new ideas or perspectives that made you rethink stuff?  
- Did your opinion or plans change at all? If so, why?  
- What did you learn about yourself, others, or the topic itself?  
- How might this convo affect your choices or what you care about going forward?  
- Were there any moments that stood out and are worth remembering?

Just keep it real with yourself—be honest and reflective. The point here is to document how the convo made you think differently, so you can look back later and see how your views or goals evolved. Don’t just summarize what was said, but focus on how it made you grow. Keep it around 1024 characters.
"""

NEUTRAL = {
    "mirostat": 2,
    "mirostat_tau": 4, 
    "num_predict": 256,
    "mirostat_eta": 0.1, 
    "num_ctx": 4000,
    "repeat_penalty": 1.5,
    "presence_penalty": 0,
    "stop": ["\n"]
}

BIAISED = {
    "mirostat": 2,
    "mirostat_tau": 9, 
    "num_predict": 256,
    "mirostat_eta": 0.1, 
    "num_ctx": 4000,
    "repeat_penalty": 1.5,
    "presence_penalty": 1.5,
    "stop": ["\n"]
}

class Contextualizer():
    def __init__(self, model):
        self.model = model

    async def neutral_context(self, messages, bot_context):
        msgs = '\n'.join([f"{msg}" for msg in messages])
        
        system = f"""
        {bot_context}
        
        {neutral_base}
        """
        
        prompt = f"""
        The transcript to write about immediately:
        {msgs}
        """
        
        if messages:
            response = await ollama.AsyncClient().generate(
                model=self.model,
                system=system,
                prompt=prompt,
                options=NEUTRAL
            )
            return response['response']
        
        return "Reading the discord conversation, I can observe that there is no messages at the moment. I should consider sparking a new topic."
        
    async def reflection(self, messages, agent_base_prompt):
        msgs = '\n'.join([f"{msg}" for msg in messages])
        
        system = f"""
        {agent_base_prompt}
        
        Instructions for: 
        {engaged_base}
        """
        
        prompt = f"""
        The transcript to reflection about write about:
        {msgs}
        """

        response = await ollama.AsyncClient().generate(
            model=self.model,
            prompt=prompt,
            system=system,
            options=BIAISED
        )
        
        return response['response']
