import asyncio
import logging
from contextlib import contextmanager
from datetime import datetime
from collections import deque
from utils.utils import *
import modules.Memories as db
import random
import os
import pickle
import time
from collections import defaultdict
from enum import Enum, auto
from modules.Contextualizer import Contextualizer
from modules.QueryEngine import QueryEngine
from modules.Planner import Planner
from modules.Responder import Responder
from modules.WebBrowser import WebBrowser
import utils.utils as utils
from utils.prompt_generator import generate_agent_prompt

class AgentState(Enum):
    IDLE = auto()
    PROCESSING = auto()
    READ_ONLY = auto()
    INITIATING_TOPIC = auto()
    
class Logger:
    def __init__(self, persistance_id, save_logs=False):
        log_level = logging.INFO if save_logs else logging.WARNING
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f"logs/{persistance_id}_agent.log")
            ]
        )
        self.logger = logging.getLogger(persistance_id)
        self.logs = defaultdict(list)

    def log_event(self, key, input_data, output_data):
        self.logs[key].append({'input': input_data, 'output': output_data})
        self.logger.info(f"Log Key: {key} | Input: {input_data} | Output: {output_data}")

    def save_logs(self, persistance_id):
        os.makedirs("logs", exist_ok=True)
        file_path = os.path.join("logs", f"{persistance_id}_log.pkl")
        with open(file_path, "wb") as f:
            pickle.dump(self.logs, f)
        self.logger.info(f"Saved logs to {file_path}")

class Agent:
    def __init__(self, user_id, agent_conf, server, archetype):
        self.user_id = int(user_id)
        self.archetype = archetype
        self.agent_conf = agent_conf
        self.server = server

        self._initialize_components()

    # INIT
    
    def _initialize_components(self):
        archetype_conf = self._load_archetype_config(self.archetype)
        self.config = self._load_agent_config(self.agent_conf)
        
        self.persistance_id = self._get_persistence_id(self.archetype)
        self.name = archetype_conf.name
        self.monitoring_channel = self.config.channel_id
        self.plan = "No specific plan at the moment. I am simply responding."
        self.memory = self._initialize_memory()

        self.responses = asyncio.Queue()
        self.processed_messages = asyncio.Queue()
        self.event_queue = asyncio.Queue()
        self.last_discussion_time = 0
        self.read_only = False
        self.sequential = self.config.sequential_mode
        self._running = True
        self.state = AgentState.IDLE
        self.lock_queue = False

        self.responder = Responder(self.config.model)
        self.query_engine = QueryEngine(self.config.model)
        self.planner = Planner(self.config.model)
        self.contextualizer = Contextualizer(self.config.model)
        self.web_browser = WebBrowser(use_ollama=True, model=self.config.model)

        self.logger = Logger(self.persistance_id, self.config.save_logs)
        self.personnality_prompt = generate_agent_prompt(self.archetype, archetype_conf)

    def _load_archetype_config(self, archetype):
        return utils.DictToAttribute(**utils.load_yaml('archetypes.yaml')['agent_archetypes'][archetype])

    def _load_agent_config(self, agent_conf):
        return utils.DictToAttribute(**utils.load_yaml(agent_conf)['config'])

    def _get_persistence_id(self, archetype):
        persistence_prefix = self.config.persitance_prefix or ""
        return f"{persistence_prefix}_{archetype}"

    def _initialize_memory(self):
        return db.Memories(collection_name=f'{self.persistance_id}_mem.pkl')

    # INIT
    
    def stop(self):
        self._running = False

    async def add_event(self, event):
        channel_id, author_id, _, _ = event
        # event.channel_id, event.message.author.id, event.message.author.display_name, event.message.content
        
        if author_id != self.user_id and self.monitoring_channel == channel_id and not self.lock_queue:
            await self.event_queue.put(event)

    
    # --- Agent to modules helpers
    
    def get_bot_context(self):
        return f"Your name is {self.name}. It is {datetime.now():%Y-%m-%d %H:%M:%S}. You are currently on discord reading the channel {self.server.get_channel(self.monitoring_channel)['name']}"
    
    async def get_channel_context(self, channel_id, bot_context):
        msgs = [msg for msg in self.server.get_messages(channel_id).copy()]
        neutral_ctx = await self.contextualizer.neutral_context(msgs, bot_context)
        self.logger.log_event('neutral_ctxs', (msgs, bot_context), neutral_ctx)
        return neutral_ctx

    async def get_neutral_queries(self, channel_id):
        msgs = [msg for msg in self.server.get_messages(channel_id).copy()]
        context_queries = await self.query_engine.context_query(msgs)
        self.logger.log_event('context_queries', msgs, context_queries)
        return context_queries

    async def get_memories(self, plan, context, messages):
        queries = await self.query_engine.response_queries(plan, context, self.personnality_prompt, messages)
        self.logger.log_event('response_queries', (plan, context, self.personnality_prompt, messages), queries)
        memories = self.memory.query_multiple(queries)
        self.logger.log_event('memories', queries, memories)
        return memories

    async def get_response(self, plan, context, memories, messages, base_prompt):
        response = await self.responder.respond(plan, context, memories, messages, base_prompt)
        self.logger.log_event('response', (plan, context, memories, messages, base_prompt), response)
        return response

    async def get_reflection(self, messages, personality_prompt):
        reflection = await self.contextualizer.reflection(messages, personality_prompt)
        self.logger.log_event('reflections', (messages, personality_prompt), reflection)
        return reflection

    async def get_plan(self, former_plan, context, unique_memories,channel_context, base_prompt):
        plan = await self.planner.refine_plan(former_plan, context, unique_memories,channel_context, base_prompt)
        self.logger.log_event('plans', (former_plan, context, unique_memories, base_prompt), plan)
        return plan,

    async def get_new_topic(self, plan, base_prompt):
        return await self.responder.new_discussion(plan, base_prompt)

    
    # --- Routines
    
    # ------- Response Routine
    
    async def respond_routine(self):
        idle_threshold = 180
        last_active = time.time()

        while self._running:
            now = time.time()

            if self.sequential:
                if self.event_queue.qsize() > 0:
                    self.state == AgentState.PROCESSING
                    await self._process_sequentially()
                else:
                    self.state = AgentState.IDLE
            else: 
                
                if random.random() < 0.05:
                    self.lock_queue = True
                    await self._read_only()
                    self.monitoring_channel = random.choice([id for id in self.server.channels.keys() if id != self.monitoring_channel])
                    self.lock_queue = False
                
                if self.event_queue.qsize() > 0:
                    self.state = AgentState.READ_ONLY if self.read_only else AgentState.PROCESSING
                elif now - last_active > idle_threshold:
                    self.state = AgentState.INITIATING_TOPIC
                else:
                    self.state = AgentState.IDLE

                if self.state == AgentState.PROCESSING:
                    substate = random.choices(['SEQUENTIAL', 'BATCH', 'IGNORE'],weights=[0.3, 0.3, 0.4],k=1)[0]

                    if substate == 'SEQUENTIAL':
                        await self._process_sequentially()
                    elif substate == 'BATCH':
                        await self._process_batch()
                    elif substate == 'IGNORE':
                        await asyncio.sleep(1)

                    last_active = time.time()

                elif self.state == AgentState.READ_ONLY:
                    await self._read_only()

                elif self.state == AgentState.INITIATING_TOPIC and not self.read_only:
                    last_message_is_me = self.server.channels[self.monitoring_channel]['last_id'] != self.user_id
                    if last_message_is_me:
                        continue

                    last_active = time.time()
                    topic = await self.get_new_topic(self.plan, self.personnality_prompt)
                    if topic:
                        await self.responses.put((topic, self.monitoring_channel))

            await asyncio.sleep(random.uniform(0, self.config.max_random_response_delay))
            await asyncio.sleep(self.config.response_delay)
            
    async def _process_messages(self, messages):
        
        formatted_messages = [self.server.format_message(author_id, global_name, content) for _, author_id, global_name, content in messages]

        context = await self.get_channel_context(self.monitoring_channel, self.get_bot_context())
        memories = await self.get_memories(self.plan, context, formatted_messages)
        response = await self.get_response(self.plan, context, memories, formatted_messages, self.personnality_prompt)
        
        if response:
            await self.responses.put((response, messages[0][0]))
            
        for message in formatted_messages:
            await self.processed_messages.put(message)
        

    async def _process_sequentially(self):
        channel_id, author_id, global_name, content = await self.event_queue.get()
        await self._process_messages([(channel_id, author_id, global_name, content)])

    async def _process_batch(self):
        batch = []
        while not self.event_queue.empty():
            channel_id, author_id, global_name, content = await self.event_queue.get()
            batch.append((channel_id, author_id, global_name, content))
        
        if batch:
            await self._process_messages(batch)

    async def _read_only(self):
        if not self.event_queue.empty():
            channel_id, author_id, global_name, content = await self.event_queue.get()
            message = self.server.format_message(author_id, global_name, content)
            await self.processed_messages.put(message)

    # ------- Plan Routine
    
    async def plan_routine(self):
        while self._running and self.config.plan_interval != -1:
            await asyncio.sleep(self.config.plan_interval)
            context = await self.get_channel_context(self.monitoring_channel, self.get_bot_context())
            neutral_queries = await self.get_neutral_queries(self.monitoring_channel)
            memories = self.memory.query_multiple(neutral_queries)
            channel_context = await self.get_channel_context(self.monitoring_channel, self.get_bot_context())
            updated_plan = await self.get_plan(self.plan, context, memories, channel_context, self.personnality_prompt)
            self.plan = updated_plan if updated_plan != None else self.plan
                
    # ------- Memory Routine

    async def memory_routine(self):
        while self._running and self.config.memory_interval != -1:
            await asyncio.sleep(self.config.memory_interval)
            messages = [await self.processed_messages.get() for _ in range(self.processed_messages.qsize())]
            if messages:
                reflection = await self.get_reflection(messages, self.personnality_prompt)
                self.memory.add_document(reflection, 'MEMORY')