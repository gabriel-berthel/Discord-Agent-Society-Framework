import ollama
from utils.utils import *

neutral_base = """
You are a student summarizing a Discord conversation. 
Your goal is to create a clear and neutral summary, using first-person language for your contributions (when your name appear).

Focus on:
- Key points and decisions made in the conversation.
- Names of people, companies, events, or any identifiable entities.
- Keep the tone objective and factual—avoid opinions or analysis.

Start with: "Reading the Discord conversation, I can observe that..."  and write a paragraphe. Keep the summary concise and fact-based.
"""

engaged_base = """
Imagine you’re jotting down a quick reflective note after a chat, like you’re talking to yourself on Discord. This is where you get to think about how the convo shifted your mindset or changed your vibe.

Here’s what to think about: Think about what we discussed—what stood out to you the most, and why. 
Were there any ideas or perspectives that shifted how you see things? 
Maybe your opinion or plans changed along the way—if so, what sparked that shift? 
Consider what you learned, whether it’s about yourself, others, or the topic itself. 
ZHow might this conversation shape the choices you make or the values you hold moving forward? And finally, were there any moments that felt especially meaningful or worth holding onto?

Start with "I feel that..." or "I noticed that..." and write a paragraphe.

Just keep it real with yourself—be honest and reflective.  
"""

NEUTRAL = {
    "mirostat": 2,
    "mirostat_tau": 4, 
    "num_predict": 300,
    "mirostat_eta": 0.1, 
    "num_ctx": 2048,
    "repeat_penalty": 0,
    "presence_penalty": 0,
    "frequency_penalty": 0.7,
    "stop": ["<|endoftext|>"],
    "newline_penalty": True
}

BIAISED = {
    "mirostat": 2,
    "mirostat_tau": 10, 
    "num_predict": 400,
    "mirostat_eta": 0.1, 
    "num_ctx": 300,
    "num_ctx": 4096,
    "repeat_penalty": 1.5,
    "presence_penalty": 1.5,
    "frequency_penalty": 0.2,
    "stop": ["<|endoftext|>"],
    "newline_penalty": True
}

class Contextualizer():
    def __init__(self, model):
        self.model = model

    async def neutral_context(self, messages, bot_context):
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
            response = await ollama.AsyncClient().generate(
                model=self.model,
                system=system,
                prompt=prompt,
                options=NEUTRAL
            )
            return clean_response(response['response'])
        
        return "Reading the discord conversation, I can observe that there is no messages at the moment. I should consider sparking a new topic."
        
    async def reflection(self, messages, agent_base_prompt):
        msgs = '\n'.join([f"{msg}" for msg in messages])
        
        system = f"""
        {agent_base_prompt}
        """
        
        prompt = f"""
        {engaged_base}
        
        
        Based on your personality, here is transcript to reflection about:
        {msgs}
        """

        response = await ollama.AsyncClient().generate(
            model=self.model,
            prompt=prompt,
            system=system,
            options=BIAISED
        )
        
        return clean_response(response['response'])
