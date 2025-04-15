import asyncio
from contextlib import contextmanager
from datetime import datetime
from collections import deque
from utils.utils import *
import modules.Memories as db
import os
import pickle
import random
from collections import defaultdict
from modules.Contextualizer import Contextualizer
from modules.QueryEngine import QueryEngine
from modules.Planner import Planner
from modules.Responder import Responder
from modules.WebBrowser import WebBrowser
import utils.utils as utils
from utils.prompt_generator import generate_agent_prompt

heading_off_messages = [
    "I'm heading off now, take care!",
    "Alright, I'm off. See you later!",
    "Time to go! Catch you soon.",
    "I need to head out, see you later!",
    "I'm about to leave, talk soon!",
    "Gotta run! Take care!",
    "Off I go, bye for now!",
    "I’m off, stay safe!",
    "It's time for me to go. Bye for now!",
    "I'm heading out, have a good one!",
    "Alright, that's my cue to leave. Catch you later!",
    "I’ve got to take off, see you later!",
    "I’m out, talk to you soon!",
    "Time for me to head off. Take care of yourself!",
    "I need to go, but let’s chat again soon!",
    "I’m out the door, catch you on the flip side!",
        "What's up, gamers?",
    "Yo yo yo, who's online?",
    "Sup nerds 😎",
    "Ayo, what’s the drama today?",
    "Who woke up and chose chaos?",
    "Good day to start some sh*t, huh?",
    "Rise and grind, pixel warriors.",
    "Reporting live from my bed.",
    "What are we screaming about today?",
    "Knock knock, it's me, your problem.",
    "Hey losers (affectionate)",
    "Let’s get this bread—or cry trying.",
    "Just vibing. You?",
    "Guess who just logged on with zero context?",
    "Hey, any lore updates?",
    "Back from the void. What did I miss?"
]

greetings = [
    "Hi there!",
    "Hello!",
    "Hey!",
    "What's up?",
    "Howdy!",
    "Hey there, how’s it going?",
    "Hi! How are you?",
    "Greetings!",
    "Hey, what's going on?",
    "Hello, how's everything?",
    "Yo!",
    "Hey, how's it going?",
    "Hi, hope you're doing well!",
    "Salutations!",
    "What’s up? How’s your day going?",
    "Hey, long time no see!",
        "Alright, I'm logging off before I say something cursed.",
    "Peace out, I’ve caused enough chaos for one day.",
    "Time to disappear like my responsibilities.",
    "Vanishing like a bad WiFi signal. Bye!",
    "Catch y’all on the flip flop.",
    "I’m off to touch grass. Maybe.",
    "Logging out before my brain fully melts.",
    "That’s enough internet for today. Bye!",
    "Alright, I’m ghosting the server now. Later!",
    "Yeeting myself into the offline zone.",
    "Time to clock out of existence. Peace!",
    "I'm outtie like a 90s kid. Bye!",
    "Later nerds, don't burn the place down.",
    "Off to go scream into the void, brb never.",
    "Catch me in the next episode of whatever this is.",
    "Dipping like chips in salsa. See ya!"
]

class Agent:
    def __init__(self, user_id, agent_conf, server, archetype):
        archetype_conf = utils.DictToAttribute(**utils.load_yaml('archetypes.yaml')['agent_archetypes'][archetype])
        self.config = utils.DictToAttribute(**utils.load_yaml(agent_conf)['config'])

        persitance_prefix = self.config.persitance_prefix if self.config.persitance_prefix else ""
        self.persistance_id = f"{persitance_prefix}_{archetype}_{user_id}"

        self.user_id = int(user_id)
        self.name = archetype_conf.name
        self.monitoring_channel = self.config.channel_id
        self.plan = "No specific plan at the moment. I am simply responding."
        self.memory = db.Memories(collection_name=self.persistance_id)
        self.responses: asyncio.Queue = asyncio.Queue()
        self.server = server
        self.processed_messages = asyncio.Queue()
        self.event_queue = asyncio.Queue()
        
        self.personnality_prompt, self.guideline = generate_agent_prompt(archetype, archetype_conf)
        
        self.log = self.config.save_logs
        self.is_online = True
        self.impulses = self.config.impulses
        self.sequential = self.config.sequential_mode
        self._running = True 

        self.logs = {
            'plans': [],
            'reflections': [],
            'context_queries': [],
            'neutral_ctxs': [],
            'response_queries': [],
            'memories': [],
            'summuries': [],
            'web_queries': []
        }

    def stop(self):
        self._running = False
        
    def save_logs(self):
        os.makedirs("logs", exist_ok=True)

        file_path = os.path.join("logs", f"{self.persistance_id}_log.pkl")

        with open(file_path, "wb") as f:
            pickle.dump(self.logs, f)

        print(f"[LOG] Saved logs to {file_path}")    

    async def add_event(self, event):
        if self.is_online:
            await self.event_queue.put(event)

    def get_bot_context(self):
        return f"You are reading {self.server.get_channel(self.monitoring_channel)['name']} and it is {datetime.now():%Y-%m-%d %H:%M:%S}"

    async def get_channel_context(self, channel_id, bot_context):
        msgs = [msg for msg in self.server.get_messages(channel_id).copy()]
        neutral_ctx = await Contextualizer(self.config.model).neutral_context(msgs, bot_context)

        if self.log:
            self.logs['neutral_ctxs'].append({'input': (msgs, bot_context), 'ouput': neutral_ctx})

        return neutral_ctx

    async def get_neutral_queries(self, channel_id):
        msgs = [msg for msg in self.server.get_messages(channel_id).copy()]
        context_queries = await QueryEngine(self.config.model).context_query(msgs)

        if self.log:
            self.logs['context_queries'].append({'input': msgs, 'ouput': context_queries})

        return context_queries

    async def get_memories(self, plan, context, messages):
        queries = await QueryEngine(self.config.model).response_queries(plan, context, messages)
        memories = self.memory.query_multiple(queries)

        if self.log:
            self.logs['response_queries'].append({'input': (plan, context, messages), 'ouput': queries})
            self.logs['memories'].append({'input': queries, 'ouput': memories})

        return memories

    async def get_search(self, plan, context, messages):
        queries = await QueryEngine(self.config.model).web_queries(plan, context, messages)
        
        summury = ""
        try:
            search = await WebBrowser(True).summarize_search(queries, 1024)
            summury = search['summary']
            if self.log:
                self.logs['web_queries'].append({'input': (plan, context, messages), 'ouput': queries})
                self.logs['summuries'].append({'input': queries, 'ouput': summury})

        except Exception as e:
            pass
    
        return summury

    async def get_response(self, plan, context, memories, messages, base_prompt):
        response = await Responder(self.config.model).respond(plan, context, memories, messages, base_prompt)

        if self.log:
            self.logs.setdefault('responses', []).append({
                'input': (plan, context, memories, messages, base_prompt),
                'ouput': response
            })

        return response

    async def get_reflection(self, messages, bot_context, personality_prompt):
        reflection = await Contextualizer(self.config.model).reflection(messages, bot_context, personality_prompt)

        if self.log:
            self.logs['reflections'].append({'input': (messages, bot_context, personality_prompt), 'output': reflection})

        return reflection

    async def get_plan(self, former_plan, context, unique_memories, bot_context, base_prompt):
        plan = await Planner(self.config.model).refine_plan(former_plan, context, unique_memories, bot_context, base_prompt)

        if self.log:
            self.logs['plans'].append({'input': (former_plan, context, unique_memories, bot_context, base_prompt), 'output': plan})

        return plan

    async def get_new_topic(self, plan, base_prompt):
        return await Responder(self.config.model).new_discussion(plan, base_prompt)

    async def respond_routine(self):
        while self._running:
            if self.event_queue.qsize() > 0 and self.is_online:
                context = await self.get_channel_context(self.monitoring_channel, self.get_bot_context())

                if not self.sequential:
                    messages = [self.server.format_message(*(await self.event_queue.get()), self.user_id)
                                for _ in range(self.event_queue.qsize())]
                else:
                    _, author_id, global_name, content = await self.event_queue.get()
                    messages = [self.server.format_message(author_id, global_name, content, self.user_id)]

                for message in messages:
                    await self.processed_messages.put(message)

                memories = await self.get_memories(self.plan, context, messages)
                response = await self.get_response(self.plan, context, memories, messages, self.personnality_prompt + self.guideline)

                if response:
                    await self.responses.put((response, self.monitoring_channel))

            await asyncio.sleep(self.config.message_throttle)
            await asyncio.sleep(random.uniform(0, self.config.max_random_msg_sleep))

    async def plan_routine(self):
        while self._running and self.config.plan_interval != -1:
            await asyncio.sleep(self.config.plan_interval)

            context = await self.get_channel_context(self.monitoring_channel, self.get_bot_context())
            neutral_queries = await self.get_neutral_queries(self.monitoring_channel)
            memories = self.memory.query_multiple(neutral_queries)

            updated_plan = await self.get_plan(self.plan, context, memories, self.get_bot_context(), self.personnality_prompt)

            if updated_plan:
                self.plan = updated_plan

    async def memory_routine(self):
        while self._running and self.config.memory_interval != -1:
            await asyncio.sleep(self.config.memory_interval)
            messages = [await self.processed_messages.get() for _ in range(self.processed_messages.qsize())]
            if messages:
                reflection = await self.get_reflection(messages, self.get_bot_context(), self.personnality_prompt)
                self.memory.add_document(reflection, 'MEMORY')

    async def impulse_routine(self):
        while self._running and self.impulses:
            action_choice = random.choices(['switch_channel', 'log_off', 'log_on', 'new_topic'])[0]

            if action_choice == 'switch_channel':
                self.is_online = False
                await asyncio.sleep(self.config.message_throttle)
                self.event_queue.empty()
                self.monitoring_channel = random.choice(list(self.server.channels.keys()))
                self.is_online = True
                sleep_time = random.uniform(20, 60 * 5)

            elif action_choice == 'log_off':
                await self.responses.put((random.choice(heading_off_messages), self.monitoring_channel))
                self.is_online = False
                await asyncio.sleep(self.config.message_throttle)
                self.event_queue.empty()
                sleep_time = random.uniform(60, 60 * 60)

            elif action_choice == 'log_on':
                await self.responses.put((random.choice(greetings), self.monitoring_channel))
                self.is_online = True
                sleep_time = random.uniform(60, 60 * 60)

            else:
                msg = await self.get_new_topic(self.plan, self.personnality_prompt + self.guideline)
                await self.responses.put((msg, self.monitoring_channel))
                sleep_time = random.uniform(60, 60 * 5)

            await asyncio.sleep(sleep_time)
