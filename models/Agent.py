import asyncio
import logging
from datetime import datetime
from utils.utils import *
import modules.agent_memories as db
import random
import time
from collections import deque
from enum import Enum, auto
from modules.agent_summuries import Contextualizer
from modules.query_engine import QueryEngine
from modules.agent_planner import Planner
from modules.agent_response_handler import Responder
import utils.utils as utils
from utils.prompt_generator import generate_agent_prompt
import math
from models.agent_logger import AgentLogger
from models.agent_state import AgentState
from models.archetype import ArchetypeManager
from models.discord_server import DiscordServer

class Agent:
    def __init__(self, user_id, agent_conf, server, archetype):
        self.user_id: int = int(user_id)
        self.archetype: str = archetype
        self.agent_conf: str = agent_conf
        self.server: DiscordServer = server
        
        # .yaml config related attributes
        self.config = utils.DictToAttribute(**utils.load_yaml(self.agent_conf)['config'])
        self.monitoring_channel: int = self.config.channel_id
        self.persistance_prefix: str = self.config.persitance_prefix 
        self.persistance_id: str = f"{self.persistance_prefix}_{self.archetype}" if self.persitance_prefix else ""
        self.plan: str = self.config.base_plan or "Responding to every message."
        self.sequential: bool = self.config.sequential_mode    
            
        # Logger
        self.logger = AgentLogger(self.persistance_id, self.config.log_path, self.config.save_logs)
        self.logger.logger.info(f"Agent-Info: [key={self.name}] | Agent Config loaded")
        
        # archetype.yaml related attributes
        self.archetype_conf =  utils.DictToAttribute(**utils.load_yaml('archetypes.yaml')['agent_archetypes'][self.archetype])
        self.personnality_prompt: str = generate_agent_prompt(self.archetype, self.archetype_conf)
        self.name:str = self.archetype_conf.name
        
        # Agent Queues
        self.responses: asyncio.Queue = asyncio.Queue()
        self.processed_messages: asyncio.Queue  = asyncio.Queue()
        self.event_queue: asyncio.Queue  = asyncio.Queue()
        self.last_messages: deque = deque(maxlen=5)
        self.logger.logger.info(f"Agent-Info: [key={self.name}] | Queue created")
        
        # Agent State variable
        self.last_discussion_time = 0
        self.read_only: bool = False
        self._running: bool = True
        self.state: AgentState = AgentState.IDLE
        self.lock_queue: bool = False
        self.memory_count: int = 0
        self.logger.logger.info(f"Agent-Info: [key={self.name}] | State variable loaded")
        
        # Agent Modules
        self.responder = Responder(self.config.model)
        self.query_engine = QueryEngine(self.config.model)
        self.planner = Planner(self.config.model)
        self.contextualizer = Contextualizer(self.config.model)
        self.memory = db.Memories(collection_name=f'{self.persistance_id}_mem.pkl', base_folder=self.config.persistance_path)
        self.logger.logger.info(f"Agent-Info: [key={self.name}] | Module Loaded")
        
    # --- MISC ---
    
    def stop(self) -> None:
        """Stops agent modules at next iteration"""
        self._running = False

    async def add_event(self, event) -> None:
        """Add event to agent event queue"""
        channel_id, author_id, _, _ = event
        
        # If queue is not locked, and agent is monitoring the channel
        if author_id != self.user_id and self.monitoring_channel == channel_id and not self.lock_queue:
            await self.event_queue.put(event)

    # --- Module Helper ---
    
    def get_bot_context(self) -> str:
        """Returns some information about what the bot is doing. Ie... time, name, channel being monitored."""
        return f"Your name is {self.name}. It is {datetime.now():%Y-%m-%d %H:%M:%S}. You are currently on discord reading the channel {self.server.get_channel(self.monitoring_channel)['name']}"
    
    async def get_channel_context(self, channel_id, bot_context) -> str:
        """
        Gets the deque from channel message queue and write a neutral summury about it.
        This helps agents not acting like godfishes.
        """
        msgs = [msg for msg in self.server.get_messages(channel_id).copy()]
        neutral_ctx = await self.contextualizer.neutral_context(msgs, bot_context)
        self.logger.log_event('neutral_ctxs', (msgs, bot_context), neutral_ctx)
        return neutral_ctx

    async def get_neutral_queries(self, channel_id) -> list[str]:
        """
        Create queries later used to retrive memories. 
        These queries are "neutral" as only the dialogues are passed. No plan, memories or base prompt.
        """
        msgs = [msg for msg in self.server.get_messages(channel_id).copy()]
        context_queries = await self.query_engine.context_query(msgs)
        self.logger.log_event('context_queries', msgs, context_queries)
        return context_queries

    async def get_memories(self, plan, context, messages) -> list[str]:
        """
        Get memories from agent vector database. First Response Queries are retrived. As opposed to neutral queires, theses
        are creating using personality prompt and plan. Then these queries are used to fetch documents in the agent memories.
        """
        queries = await self.query_engine.response_queries(plan, context, self.personnality_prompt, messages)
        self.logger.log_event('response_queries', (plan, context, self.personnality_prompt, messages), queries)
        memories = self.memory.query_multiple(queries)
        self.logger.log_event('memories', queries, memories)
        return memories

    async def get_response(self, plan, context, memories, messages, base_prompt) -> str:
        """
        Prompt for a response from the response modules. These will be the final output visible by users.
        """
        response = await self.responder.respond(plan, context, memories, messages, base_prompt, self.last_messages)
        self.logger.log_event('response', (plan, context, memories, messages, base_prompt), response)
        self.last_messages.append(response)
        return response

    async def get_reflection(self, messages, personality_prompt) -> str:
        """
        Prompt for a reflection, then stored in memories.
        """
        reflection = await self.contextualizer.reflection(messages, personality_prompt)
        self.logger.log_event('reflections', (messages, personality_prompt), reflection)
        return reflection

    async def get_plan(self, former_plan, context, unique_memories,channel_context, base_prompt) -> str:
        """
        Prompt for a new plan, then stored in memory and passed to other modules.
        """
        plan = await self.planner.refine_plan(former_plan, context, unique_memories,channel_context, base_prompt)
        self.logger.log_event('plans', (former_plan, context, unique_memories, base_prompt), plan)
        return plan

    async def get_new_topic(self, plan, base_prompt) -> str:
        """
        Prompt for a spontatnous message, without giving context. Given enough time, this allow agents to develop a diverse range of memories
        instead of looping on the same subjects.
        """
        return await self.responder.new_discussion(plan, base_prompt)

    
    # --- Routines
    
    # ------- Response Routine
    
    async def respond_routine(self) -> None:
        """
        If sequential mode, the event queue is ALWAYS emptied entirely (at time of the call).
        This is used for benchmarking or more globally for the prompting client! Though it could work on servers,
        this would lead the agent to respond as soon as availible.
        
        If not sequential mode, there's random at play to make agent more "human"
        """
       
        idle_threshold = 60*5
        last_active = time.time()
        
        # Sequential mode -> Agent always respond to everything in their queue ASAP
        if self.sequential:
            while self._running:
                if self.event_queue.qsize() > 0:
                    self.state == AgentState.PROCESSING
                    await self._process_batch()
                else:
                    self.state = AgentState.IDLE
            
            await asyncio.sleep(self.config.response_delay)
            await asyncio.sleep(random.uniform(0, self.config.max_random_response_delay))
        
        # Agent will behave more randomly
        # It's not always easy to keep track of what they're doing
        # But it's good as human are not predictable! sometime they answer, sometime they read but forget, sometime they ignore...
        while self._running and not self.sequential:
            now = time.time()
            old_state = self.state
            try:
                
                # 5% chance channel switch
                # Read Only is called to gracefully empty event queue but still form memories down the line.
                # Lock is used to avoid having new events added during.
                if random.random() < 0.05:
                    self.lock_queue = True
                    await self._read_only()
                    self.monitoring_channel = random.choice([id for id in self.server.channels.keys() if id != self.monitoring_channel])
                    self.logger.logger.info(f"Agent-Channel: [key={self.name}] | Switched to channel: {self.server.get_channel(self.monitoring_channel)}")
                    self.lock_queue = False
                
                # If no event in event queue, then agent will try to initiate topic.
                # self.read_only is not really set anywhere but it's there to dynamically mute the agent if necessary.
                if self.event_queue.qsize() > 0:
                    self.state = AgentState.READ_ONLY if self.read_only else AgentState.PROCESSING
                elif now - last_active > idle_threshold:
                    self.state = AgentState.INITIATING_TOPIC
                else:
                    self.state = AgentState.IDLE
                
                if old_state != self.state:
                    self.logger.logger.info(f"Agent-State: [key={self.name}] | New State: {self.state}")

                if self.state == AgentState.PROCESSING:
                    
                    # Randomly select to ignore, only read messages or respond. Weights are random with range to make agent less predictable.
                    batch_weight = round(random.uniform(0.5, 0.7), 3)
                    ignore_weight = round(random.uniform(0.05, 0.1), 3)
                    only_read_weight = max(1 - round(batch_weight + ignore_weight, 3), 0)
                    substate = random.choices(['BATCH', 'IGNORE', 'ONLY_READ'],weights=[batch_weight, ignore_weight, only_read_weight],k=1)[0]
                    self.logger.logger.info(f"Agent-Substate: [key={self.name}] | Substate: {substate}, w=[batch_weight={batch_weight}, ignore_weight={ignore_weight}, only_read_weight={only_read_weight}]")

                    if substate == 'BATCH':
                        # Will respond to a random number of elements in the queue.
                        # No need to check queue size as _process_batch takes the min.
                        noelem=int(random.uniform(1, 10))
                        await self._process_batch(noelem)
                    elif substate == 'ONLY_READ':
                        # Will put everything in processed_message queue but not respond
                        await self._read_only()
                    elif substate == 'IGNORE':
                        # Messaged won't even be in processed_message queue.
                        await self._ignore()

                    last_active = time.time()
                elif self.state == AgentState.READ_ONLY:
                    await self._read_only()
                elif self.state == AgentState.INITIATING_TOPIC and not self.read_only:
                    # Agent will try to initiate topic though, we check the last message in the channel wasn't send by the agent.
                    last_message_is_me = self.server.channels[self.monitoring_channel]['last_id'] != self.user_id
                    if last_message_is_me:
                        self.logger.logger.info(f"Agent-Subtate: [key={self.name}] | last_message_is_me=true")
                        # though there is 0.5 % chance the agent will send a new message anyway.
                        # so every cycle, if (ie 5 minutes) without any message after this agent sent one...
                        # agent may still try to bring a new topic!
                        # this bit of random is important as can diversify conversations
                        if not random.random() < 0.005:
                            self.logger.logger.info(f"Agent-Subtate: [key={self.name}] | Aborting topic initialisation")
                            continue
                        
                    # if last message is another agent then an entirely new topic will be made.
                    # not much context is given to LLM so only the personality and plan determine the output.
                    last_active = time.time()
                    topic = await self.get_new_topic(self.plan, self.personnality_prompt)
                    if topic:
                        await self.responses.put((topic, self.monitoring_channel))
                        await self.processed_messages.put(f'[Me] {topic}')
                        self.logger.logger.info(f"Agent-Output: [key={self.name}] | Created new topic: {topic}")
                
            except Exception as e:
                self.logger.logger.error(f"Agent-Routine: [key={self.name}] | Error with responder routine: {e}")
            
            # random delays to make agent more "human"
            await asyncio.sleep(self.config.response_delay)
            await asyncio.sleep(random.uniform(0, self.config.max_random_response_delay))
            
    async def _process_messages(self, messages) -> None:
        """
        This methods process a list of message meaning:
            - It will format them properly for downstream modules
            - Retrieves every element needed to form a response
            - Append response to agent response queue if needed
            - Appends messages in agent processed message queues
            
        99% of the time this leads to a response from the agent tho, in rare occasion, the models outputs nothing.
        In this case, we simply output an empty response that can later be filtered.
        """
        
        formatted_messages = [self.server.format_message(author_id, global_name, content) for _, author_id, global_name, content in messages]

        context = await self.get_channel_context(self.monitoring_channel, self.get_bot_context())
        memories = await self.get_memories(self.plan, context, formatted_messages)
        response = await self.get_response(self.plan, context, memories, formatted_messages, self.personnality_prompt)
        
        if response:
            await self.responses.put((response, messages[0][0]))
        else: 
            await self.responses.put(("", messages[0][0]))
            
        for message in formatted_messages:
            await self.processed_messages.put(message)

    async def _process_batch(self, noelem:int=None) -> None:
        """
        This methods calls _process_messages()! If noelem is defined, it will retrive the at most noelem elements from the event queue.
        This is useful in the response routine as this allow to randomly select an amount of message to process, giving more change for random
        ignore or the agent simply reading without responding. Though, it can be used to empty the batch entirely... as used in sequential mode.
        
        """
        
        q_size = self.event_queue.qsize()
        noelem = q_size if not noelem else min(q_size, noelem)
        
        batch = [
            await self.event_queue.get() 
            for _ in range(noelem)
        ]
        
        if batch:
            self.logger.logger.info(f"Agent-Info: [key={self.name}] | Processed {noelem} elements in the {q_size} events from the event queue")
            await self._process_messages(batch)
            
    async def _read_only(self) -> None:
        """
        As opposed to _process_batch, this will read the entire queue and put every message in processed queue.
        No responder module is called. This allow agent to form memories from message without having to respond.
        
        This is useful to simulate human behavior but also swich channel, gracefully emptying the event queue.
        """
        current_size = self.event_queue.qsize()
        for _ in range(current_size):
            channel_id, author_id, global_name, content = await self.event_queue.get()
            message = self.server.format_message(author_id, global_name, content)
            self.logger.logger.info(f"Agent-Info: [key={self.name}] | Processing message from {global_name} (read-only)")
            await self.processed_messages.put(message)

    async def _ignore(self) -> None:
        """
        This will ignore one message from the event queue. Apart from the context summury, agent will never process this one.
        """
        if not self.event_queue.empty():
            await self.event_queue.get()
            self.logger.logger.info(f"Agent-Info: [key={self.name}] | Ignoring Message in event queue")


    # ------- Plan Routine
    
    async def plan_routine(self) -> None:
        """
        This methods works with the memory_count variable.
        If plans are activated, plans are formed every 5 new memories. Modulo 6 as plans are added into memories.
        """
        while self._running and self.config.plans:
            try:
                self.logger.logger.info(f"Agent-Routine: [key={self.name}] | Started plan routine")
                if self.memory_count % 6 == 0:
                    context = await self.get_channel_context(self.monitoring_channel, self.get_bot_context())
                    neutral_queries = await self.get_neutral_queries(self.monitoring_channel)
                    memories = self.memory.query_multiple(neutral_queries)
                    channel_context = await self.get_channel_context(self.monitoring_channel, self.get_bot_context())
                    updated_plan = await self.get_plan(self.plan, context, memories, channel_context, self.personnality_prompt)
                    self.plan = updated_plan if updated_plan != None else self.plan
                    
                    if updated_plan:
                        self.memory_count += 1
                        self.memory.add_document(updated_plan, 'PLAN')
                        self.logger.logger.info(f"Agent-Routine: [key={self.name}] | Updated plan")
                else:
                    self.logger.logger.info(f"Agent-Routine: [key={self.name}] | Not enough memories to change plan")
                    
            except Exception as e:
                self.logger.logger.error(f"Agent-Routine: [key={self.name}] | Error with planning routine: {e}")
                
            await asyncio.sleep(30)
                
    # ------- Memory Routine

    async def memory_routine(self) -> None:
        """
        This methods works with the processed_message queue.
        If memories (reflection) are activated, they are formed every 5 message processed.
        _read_only() no makes more senses as it does not call the responder but put the message in the response queue.
        """
        while self._running and self.config.memories:
            try:
                self.logger.logger.info(f"Agent-Routine: [key={self.name}] | Starting memory routine")
                if self.processed_messages.qsize() >= 5:
                    messages = [await self.processed_messages.get() for _ in range(5)]
                    if messages:
                        reflection = await self.get_reflection(messages, self.personnality_prompt)
                        self.memory.add_document(reflection, 'MEMORY')
                        self.memory_count += 1
                        self.logger.logger.info(f"Agent-Routine: [key={self.name}] | Created memory")
                else:
                    self.logger.logger.info(f"Agent-Routine: [key={self.name}] | Not enough message to process memories")
            except Exception as e:
                self.logger.logger.error(f"Agent-Routine: [key={self.name}] | Error with memory routine: {e}")
            
            await asyncio.sleep(30)