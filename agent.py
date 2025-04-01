import asyncio
from contextlib import contextmanager
import hikari
import os
import random
import time
from datetime import datetime
import ollama
from collections import deque
from utils import *
import entities.Memories as db
from collections import defaultdict
from modules.Contextualizer import Contextualizer
from modules.QueryEngine import QueryEngine
from modules.Planner import Planner
from modules.Responder import Responder

class Agent:
    def __init__(self, user_id, agent_conf, server):
        self.user_id = int(user_id)
        self.config = agent_conf['config']
        self.model = agent_conf['config']['model']
        self.monitoring_channel = 1350127264871743498
        self.plan = "Observe and wait."
        self.messages = deque()
        self.memory = db.Memories(collection_name="AgentMemTest")
        self.responses: asyncio.Queue = asyncio.Queue()
        self.event_queue = asyncio.Queue()
        self.notification_queue = asyncio.Queue()
        self.last_choice = deque(maxlen=10)
        self.server = server
        self.processed_messages = asyncio.Queue()
    
    def get_bot_context(self):
        return f"You are reading {self.server.get_channel(self.monitoring_channel)['name']} and it is {datetime.now():%Y-%m-%d %H:%M:%S}"

    async def get_channel_context(self, channel_id):
        return await Contextualizer(self.model).neutral_context([msg for msg in self.server.get_messages(channel_id).copy()], self.get_bot_context())
    
    async def get_channel_queries(self, channel_id):
        return await QueryEngine(self.model).context_query([msg for msg in self.server.get_messages(channel_id).copy()])

    # ----- SERVER ROUTINES juste 3 agents & 2 salons pour la preuve de concept! Sinon benchmarker va être horrible.
    # Separate modules.. Contextualizer
    # It would be nice for query to able to do web-serch
    async def respond_routine(self):
        while True:
            if self.event_queue.qsize() > 0:
                context = await self.get_channel_context(self.monitoring_channel)
                messages = [self.server.format_message(await self.event_queue.get()) for _ in range(self.event_queue.qsize())]
                
                queries = await QueryEngine(self.model).response_queries(self.plan, context, messages)
                memories = self.memory.query_multiple(queries)
                
                # web_queries = await QueryEngine(self.model).web_queries(self.plan, context, messages)
                # summary = .... web browser summary
                # putting summary in memory

                response = await Responder(self.model).respond(self.plan, context, memories, messages)
                
                for message in messages:
                    await self.processed_messages.put(message)

                if response:
                    await self.responses.put((response, self.monitoring_channel))

            await asyncio.sleep(self.config['message_throttle'])

    async def plan_routine(self):
        while True:
            await asyncio.sleep(self.config['plan_interval'])
            print("Starting Plan Routine")
            
            context = await self.get_channel_context(self.monitoring_channel)
            channel_queries = await self.get_channel_queries(self.monitoring_channel)

            memories = self.memory.query_multiple(channel_queries + [self.plan])
            most_recent_mem = self.memory.get_last_n_memories(3)
            unique_memories = list(set(memories + most_recent_mem))

            choice_num = await Planner(self.model).choose_action(self.plan, context, unique_memories)
            choice = f" keep reading {self.server.channels[self.monitoring_channel]['name']}"
            updated_plan = await Planner(self.model).refine_plan(self.plan, context, unique_memories, choice)
            
            if updated_plan:
                self.memory.add_document(self.plan, 'FORMER-PLAN')
                self.plan = updated_plan

    
    async def memory_routine(self):
        while True:
            # TODO : FIX THE QUUEUE
            await asyncio.sleep(self.config['memory_interval'])
            print("Starting Memory Routine")

            messages = [await self.processed_messages.get() for _ in range(self.processed_messages.qsize())]
            if messages:
                reflection = await Contextualizer(self.model).reflection(messages, self.get_bot_context())
                self.memory.add_document(reflection, 'MEMORY')