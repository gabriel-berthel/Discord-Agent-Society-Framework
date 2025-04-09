import ollama
from utils import *

neutral_base = """
Imagine you're a journalism student tasked with summarizing a Discord conversation you participated in. 
Your goal is to demonstrate your ability to accurately and impartially summarize the conversation while using first-person language. 
Start the summary with the following sentence: 'Reading the discord conversation, I can observe that...' 
The summary should be a long paraphrase of the conversation, adapting to the volume of information provided, and should be written as a single paragraph. 
Ensure all important details are captured. 
The tone must be neutral, impartial, and factual, maintaining clarity and coherence, just as a journalist would in reporting. 
Identify the participants by their names or identifiers and clearly outline the relationships and roles of the participants within the conversation. 
Include key insights, decisions, and discussions that occurred during the conversation. 
For any message prefaced with "You," interpret it as your message and convert it to first-person language in the summary. 
The final paragraph should serve as a factual description of the conversation, summarizing what was observed without any analysis, suggestions, or recommendations. 
The summary should focus solely on what was said and done in the conversation, reflecting your responses and interactions as noted in the transcript.
"""

engaged_base = """
Imagine you're writing a high-level reflection or memory in your electronic notebook based on the conversation you've just had. 
This is an opportunity to capture how your preferences, tastes, or perspectives may have evolved during the discussion. 
Reflect on the topics you've discussed, any insights or decisions that stood out, and how your opinions or plans may have shifted as a result. 
Consider what you’ve learned or how the conversation has influenced your thinking. 
Write this memory in a way that you can revisit later to remember how your thoughts and preferences have changed over time. 
This entry should serve as a personal reflection on your growth, based on the conversation.
"""

class Contextualizer():
    def __init__(self, model):
        self.model = model

    async def neutral_context(self, messages, bot_context):
      prompts = [('system', neutral_base)] 
      prompts += [('system', f'{bot_context}\nHere is the transcript:')]
      prompts += [('user', msg) for msg in messages]
      
      if messages:
        msg_txt = "- " + messages[0] + "\n- ".join(messages[1:])
        response = await ollama.AsyncClient().chat(
            model=self.model, 
            messages=format_llm_prompts(prompts)
        )
         
        return response['message']['content']
      
      return "Reading the discord conversation, I can observe that there is no messages at the moment. I should consider sparking a new topic."
        
    async def reflection(self, messages, bot_context, argent_base_prompt):
        prompts = [
            ('system', argent_base_prompt),
            ('system', engaged_base),
            ('system', f'{bot_context}\nHere is the transcript to reflect about:')
        ] + [('user', msg) for msg in messages]
        
        response = await ollama.AsyncClient().chat(
            model=self.model, 
            messages=format_llm_prompts(prompts)
        )
            
        return response['message']['content']