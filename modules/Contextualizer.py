import ollama
from utils import *

neutral_base = """
You are a journalism student assigned to summarize a Discord conversation in which you were an active participant. Your goal is to produce a clear, accurate, and impartial summary that demonstrates your ability to report professionally, paraphrase effectively, and preserve factual integrity.

Begin the summary with the sentence:
"Reading the Discord conversation, I can observe that..."

Your summary must follow these guidelines:

    Use first-person language to represent your own contributions. Any lines attributed to "You" in the transcript should be written in first person.
    Write the entire summary as a single paragraph, adapting its length and detail to the complexity of the conversation.
    Maintain a neutral, objective, and factual tone—as expected in professional journalism. Avoid opinions, analysis, or editorializing of any kind.

    Name every identifiable entity mentioned in the conversation, including but not limited to:

        People (e.g., full names, usernames, or provided identifiers)
        Organizations and companies (e.g., OpenAI, The New York Times, NASA)
        Creative works (e.g., books, films, TV shows, podcasts, songs, video games)
        Authors, directors, producers, artists, or creators of those works
        Tools, apps, platforms, and technologies (e.g., Discord, Zoom, Python, Google Docs)
        Projects, repositories, documents, and codebases
        Public figures, historical figures, or fictional characters
        Media outlets, websites, or publications
        Events or time references (e.g., conferences, release dates, historical events)

    Clearly identify each participant’s role, relationship, or affiliation when discernible.
    Ensure that key points, decisions, arguments, questions, and insights from the conversation are faithfully summarized.
    Be precise and avoid omitting any meaningful content or context that contributes to understanding what took place.
    The summary should reflect only what was said and done, avoiding any speculation, recommendation, or evaluation.

The final output must be a detailed, fact-based narrative that accurately captures the flow and content of the discussion, with all referenced entities explicitly named to support transparency and traceability.
The final output should be roughly 1024 caracters.
"""

engaged_base = """
Imagine you're writing a reflective entry in your personal digital notebook or memory archive, based on the conversation you’ve just had. This entry is an opportunity to capture how your preferences, beliefs, insights, or intentions may have evolved throughout the discussion.

Use this reflection to explore the following:

    What topics were discussed, and which ones stood out to you?
    What new insights, ideas, or perspectives emerged that challenged or expanded your thinking?
    Did your opinion, plan, or preference shift in any way? If so, how—and why?
    What did you learn about yourself, the topic, or others through this exchange?
    How might this conversation influence your future decisions, interests, or actions?
    What moments felt personally significant or worth remembering?

Write this memory in a tone that feels authentic to you—thoughtful, exploratory, and honest. The purpose is to document how your thinking and personal growth unfolded in response to the conversation, so that you can revisit this entry later and trace the evolution of your views, priorities, and self-awareness over time.

This is not a summary of what was said, but a reflection on how the dialogue shaped you. The final output should be roughly 1024 caracters.
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
        
    async def reflection(self, messages, bot_context, argent_base_prompt):
        prompts = [
            ('system', argent_base_prompt),
            ('system', engaged_base),
        ] + [('user', msg) for msg in messages]
        
        response = await ollama.AsyncClient().chat(
            model=self.model, 
            messages=format_llm_prompts(prompts)
        )
            
        return response['message']['content']