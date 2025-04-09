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
import ollama
from prompt_generator import generate_agent_prompt

class Agent:
    def __init__(self, user_id, agent_conf, server, archetype, special_instruction="", logs=False):
        
        self.config = utils.load_yaml(agent_conf)['config']
        self.user_id = int(user_id)
        self.monitoring_channel = self.config['initial-channel-id']
        self.plan = "No specific plan at the moment. I am simply responding."
        self.special_instruction = special_instruction
        self.memory = db.Memories(collection_name=f"{archetype}_{user_id}")
        self.responses: asyncio.Queue = asyncio.Queue()
        self.server = server
        self.processed_messages = asyncio.Queue()
        self.event_queue = asyncio.Queue()
        self.personnality_prompt = generate_agent_prompt(archetype, utils.load_yaml('archetypes.yaml')['agent_archetypes'])
        self.log = False
        
        self.logs = {
            'plans': [],
            'reflections': [],
            'context_queries': [],
            'neutral_ctxs': [],
            'response_queries': [],
            'memories': []
        }
        
    def get_bot_context(self):
        ctx =  f"You are reading {self.server.get_channel(self.monitoring_channel)['name']} and it is {datetime.now():%Y-%m-%d %H:%M:%S}"
        if self.special_instruction:
            ctx += ctx + f'\n{self.special_instruction}'
        
        return ctx

    async def get_channel_context(self, channel_id, bot_context):
        
        msgs = [msg for msg in self.server.get_messages(channel_id).copy()]
        neutral_ctx = await Contextualizer(self.config['model']).neutral_context(msgs, bot_context)
        
        if self.log:
            self.logs['neutral_ctxs'].append({
                'input': (msgs, bot_context),
                'ouput':  neutral_ctx
            })
               
        return neutral_ctx
    
    async def get_neutral_queries(self, channel_id):
        msgs = [msg for msg in self.server.get_messages(channel_id).copy()]
        context_queries = await QueryEngine(self.config['model']).context_query([msg for msg in self.server.get_messages(channel_id).copy()])
        
        if self.log:
            self.logs['context_queries'].append({
                'input': (msgs),
                'ouput':  context_queries
            })
        
        return context_queries
    
    async def get_memories(self, plan, context, messages):
        queries = await QueryEngine(self.config['model']).response_queries(plan, context, messages)                    
        memories = self.memory.query_multiple(queries)
        
        if self.log:
            self.logs['response_queries'].append({
                'input': (plan, context, messages),
                'ouput':  queries
            })
            
            self.logs['memories'].append({
                'input': (queries),
                'ouput':  memories
            })
            
        return memories
    
    async def get_response(self, plan, context, memories, messages, base_prompt):
        response = await Responder(self.config['model']).respond(plan, context, memories, messages, base_prompt)

        if self.log:
            self.logs['responses'].append({
                'input': (plan, context, memories, messages, base_prompt),
                'ouput':  response
            })

        return response
    
    async def get_reflection(self, messages, bot_context, personality_prompt):
        reflection = await Contextualizer(self.config['model']).reflection(messages, bot_context, personality_prompt)
        
        if self.log:
            self.logs['reflections'].append({
                'input': (messages, bot_context, personality_prompt),
                'output': reflection
            })
        
        return reflection
            

    async def get_plan(self, former_plan, context, unique_memories, bot_context, personnality_prompt):
        
        plan = await Planner(self.config['model']).refine_plan(former_plan, context, unique_memories, bot_context, personnality_prompt)
            
        if self.log:
            self.logs['plans'].append({
                'input':  (former_plan, context, unique_memories, bot_context, personnality_prompt),
                'output': plan
            })
        
        return plan
    
    async def respond_routine(self):
        while True:
            if self.event_queue.qsize() > 0:
                context = await self.get_channel_context(self.monitoring_channel, self.get_bot_context())
                throttle = self.config['message_throttle']
                # Batch mode: process all messages in queue
                if throttle != -1:
                    messages = [self.server.format_message(await self.event_queue.get()) 
                                for _ in range(self.event_queue.qsize())]
                # Sequential mode
                else:
                    channel_id, author_id, global_name, content = await self.event_queue.get()
                    messages = [self.server.format_message(author_id, global_name, content)]
    
                memories = await self.get_memories(self.plan, context, messages)
                response = await self.get_response(self.plan, context, memories, messages, self.personnality_prompt)
  
                for message in messages:
                    await self.processed_messages.put(message)

                if response:      
                    await self.responses.put((response, self.monitoring_channel))

            if self.config['message_throttle'] != -1:
                await asyncio.sleep(self.config['message_throttle'])
            else:
                await asyncio.sleep(1)


    async def plan_routine(self):
        while True and self.config['plan_interval'] != -1:
            await asyncio.sleep(self.config['plan_interval'])
            
            context = await self.get_channel_context(self.monitoring_channel, self.get_bot_context())
            neutral_queries = await self.get_neutral_queries(self.monitoring_channel)

            memories = self.memory.query_multiple(neutral_queries)
            most_recent_mem = self.memory.get_last_n_memories(3)
            unique_memories = list(set(memories + most_recent_mem))

            updated_plan = await self.get_plan(self.plan, context, unique_memories, self.get_bot_context(), self.personnality_prompt)

            if updated_plan:                
                self.memory.add_document(self.plan, 'FORMER-PLAN')
                self.plan = updated_plan

    async def memory_routine(self):
        while True and self.config['memory_interval'] != -1:

            await asyncio.sleep(self.config['memory_interval'])
            messages = [await self.processed_messages.get() for _ in range(self.processed_messages.qsize())]
            if messages:
                reflection = await self.get_reflection(messages, self.get_bot_context(), self.personnality_prompt)                    
                self.memory.add_document(reflection, 'MEMORY')