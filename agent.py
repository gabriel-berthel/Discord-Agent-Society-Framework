import asyncio
from contextlib import contextmanager
from datetime import datetime
from collections import deque
from utils import *
import modules.Memories as db
from collections import defaultdict
from modules.Contextualizer import Contextualizer
from modules.QueryEngine import QueryEngine
from modules.Planner import Planner
from modules.Responder import Responder
import utils

class Agent:
    def __init__(self, user_id, agent_conf, server, archetype, special_instruction=""):
        self.user_id = int(user_id)
        self.config = agent_conf.get('config')
        self.monitoring_channel = self.config.get('initial-channel-id')
        self.plan = "No specific plan at the moment. I am simply responding."
        self.special_instruction = special_instruction
        self.memory = db.Memories(collection_name="AgentMemTest")
        self.responses: asyncio.Queue = asyncio.Queue()
        self.server = server
        self.processed_messages = asyncio.Queue()
        self.event_queue = asyncio.Queue()
        self.archetype = utils.load_yaml('archetypes.yaml').get(archetype)
    
    def get_bot_context(self):
        ctx =  f"You are reading {self.server.get_channel(self.monitoring_channel)['name']} and it is {datetime.now():%Y-%m-%d %H:%M:%S}"
        if self.special_instruction:
            ctx += ctx + f'\n{self.special_instruction}'
        
        return ctx

    async def get_channel_context(self, channel_id):
        return await Contextualizer(self.config.get('model')).neutral_context([msg for msg in self.server.get_messages(channel_id).copy()], self.get_bot_context())
    
    async def get_neutral_queries(self, channel_id):
        return await QueryEngine(self.config.get('model')).context_query([msg for msg in self.server.get_messages(channel_id).copy()])

    async def respond_routine(self):
        while True:
            if self.event_queue.qsize() > 0:
                context = await self.get_channel_context(self.monitoring_channel)
                throttle = self.config.get('message_throttle', -1)

                # Batch mode: process all messages in queue
                if throttle != -1:
                    messages = [self.server.format_message(await self.event_queue.get()) 
                                for _ in range(self.event_queue.qsize())]
                # Sequential mode
                else:
                    author_id, global_name, content = await self.event_queue.get()
                    messages = [self.server.format_message(author_id, global_name, content)]

                queries = await QueryEngine(self.config.get('model')).response_queries(self.plan, context, messages)
                memories = self.memory.query_multiple(queries)

                response = await Responder(self.config.get('model')).respond(self.plan, context, memories, messages)

                for message in messages:
                    await self.processed_messages.put(message)

                if response:
                    await self.responses.put((response, self.monitoring_channel))

            # Only sleep if throttle is not -1
            if self.config.get('message_throttle', -1) != -1:
                await asyncio.sleep(self.config.get('message_throttle'))


    async def plan_routine(self):
        while True and self.config.get('plan_interval') != -1:
            await asyncio.sleep(self.config.get('plan_interval'))
            print("Starting Plan Routine")
            
            context = await self.get_channel_context(self.monitoring_channel)
            neutral_queries = await self.get_neutral_queries(self.monitoring_channel)

            memories = self.memory.query_multiple(neutral_queries)
            most_recent_mem = self.memory.get_last_n_memories(3)
            unique_memories = list(set(memories + most_recent_mem))

            updated_plan = await Planner(self.model).refine_plan(self.plan, context, unique_memories, self.get_bot_context())
            
            if updated_plan:
                self.memory.add_document(self.plan, 'FORMER-PLAN')
                self.plan = updated_plan

    async def memory_routine(self):
        while True and self.config.get('memory_interval') != -1:

            await asyncio.sleep(self.config.get('memory_interval'))
            print("Starting Memory Routine")

            messages = [await self.processed_messages.get() for _ in range(self.processed_messages.qsize())]
            if messages:
                reflection = await Contextualizer(self.model).reflection(messages, self.get_bot_context())
                self.memory.add_document(reflection, 'MEMORY')