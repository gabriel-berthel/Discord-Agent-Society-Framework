import ollama
from utils.utils import *

neutral_base = """
You are a journalism student tasked with summarizing a Discord conversation in which you participated. Your objective is to create a clear, accurate, and impartial summary, showcasing your ability to paraphrase effectively and maintain factual integrity.

Start the summary with:
"Reading the Discord conversation, I can observe that..."

Guidelines for your summary:

    Use first-person language to represent your own contributions. Any statements attributed to "Me" in the transcript should be written in the first person.

    Write the summary in a single paragraph, adjusting its length and detail based on the conversation's complexity.

    Maintain a neutral, objective, and factual tone. Avoid expressing personal opinions, analysis, or editorializing.

    Name every identifiable entity mentioned, including:

        People (full names, usernames, or identifiers)

        Organizations, companies, and platforms (e.g., OpenAI, Discord, Google Docs)

        Creative works (e.g., books, TV shows, video games)

        Tools, apps, technologies, or projects (e.g., Python, Notion, repositories)

        Public figures, historical figures, or fictional characters

        Events, conferences, or time references

    Clearly identify each participant’s role or affiliation if known.

    Summarize key points, questions, insights, and decisions, making sure to capture the flow of the conversation without omitting any crucial context.

    The summary should focus solely on what was said and done, not on personal opinions or evaluations.

The final output should be a precise, fact-based account of the conversation that accurately reflects the discussion, with all relevant entities named for transparency. Aim for a length of around 1024 characters.
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

class Contextualizer():
    def __init__(self, model):
        self.model = model

    async def neutral_context(self, messages, bot_context):
      prompts = [('system', neutral_base)] 
      prompts += [('user', msg) for msg in messages]
      
      if messages:
        msg_txt = "- " + messages[0] + "\n- ".join(messages[1:])
        response = await ollama.AsyncClient().chat(
            model=self.model, 
            messages=format_llm_prompts(prompts)
        )
         
        return response['message']['content']
      
      return "Reading the discord conversation, I can observe that there is no messages at the moment. I should consider sparking a new topic."
        
    async def reflection(self, messages, bot_context, agent_base_prompt):
        prompts = [
            ('system', agent_base_prompt),
            ('system', engaged_base),
        ] + [('user', msg) for msg in messages]
        
        response = await ollama.AsyncClient().chat(
            model=self.model, 
            messages=format_llm_prompts(prompts)
        )
            
        return response['message']['content']