import logging
import os
import random
from asyncio import Queue, sleep
from collections import deque
from datetime import datetime

import modules.agent_memories as db
from models.agent_logger import AgentLogger
from models.discord_server import DiscordServer
from modules.agent_planner import Planner
from modules.agent_response_handler import Responder
from modules.agent_summuries import Contextualizer
from modules.query_engine import QueryEngine
from utils.agent.agent_utils import *
from utils.agent.base_prompts import generate_agent_prompt
from utils.file_utils import load_yaml
from models.event import Event

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("tqdm").setLevel(logging.ERROR)
logging.getLogger("httpx").setLevel(logging.ERROR)


class Agent:
    """
    Represents an autonomous agent that processes and responds to user messages asynchronously. It integrates with various modules to:
        - Generate humanlike replies (Responder)
        - Forge Relevant Memories (Contextualizer -> reflection)
        - Summarize context for greater attention (Contextualizer -> Neutral Context)
        - Continuously refine objectives and plans (Planner)
        - Retrieve relevant memories (Memories + Query Engine)

    Contextual handling is supported through the `DiscordServer` abstraction, which maps IDs to readable names, switches channels, and caches recent messages. 
    Context summaries prevent loss of short-term context and reduce reliance on memory modules in early stages, while allowing the agent to focus on the message to process.

    Flow of operations ---

    Message flow is managed through queues:
        - `event_queue`: Consumed events from the monitored Discord channel.
        - `responses`: Output responses ready to be consumed
        - `processed_messages`: Stores messages the agent has read/handled.

    This Agent provides two processing modes for handling message streams:
        - Sequential Mode:
            Processes messages in a strict first-in, first-out (FIFO) order, producing deterministic
            responses. Suitable for rule-based or prompt-driven applications where predictability is key.

        - Non-Sequential Mode:
            Introduces random behavior in message handling to simulate human-like interactions.
            Includes skipping messages, ignoring inputs, or sending spontaneous messages. 
            It allows agent to behave unexpectedly and discover new "interest"

    Methods:
        - _read_only(event_queue): Transfers events from the `event_queue` to the `processed_messages` queue 
            without triggering the agent’s response mechanism.

        - _ignore(event_queue): Retrieves and discards a single event from the `event_queue`. No processing or response occurs.

        - _process_batch(event_queue):
        - _process_messages(event_queue):
            Processes one or more events from the `event_queue`, passing them through the full response
            pipeline. Results in updates to the `processed_messages` queue AND the `response_queue`

    Planning, memory, and responses operate asynchronously. 
    The `memory_count` and `processed_messages` queue regulate reflection/planning frequency.
        
    Some general considerations ---
    
    Archetypes must be defined in `configs/archetypes.yaml` following the same structure.
    Please refer to config documentations for further information about to be managed externally.
    """

    def __init__(self, user_id, agent_conf, server, archetype):
        self.user_id: int = int(user_id)
        self.archetype: str = archetype
        self.agent_conf: str = agent_conf
        self.server: DiscordServer = server

        # .yaml config related attributes
        self.config = DictToAttribute(**load_yaml(self.agent_conf)['config'])
        if self.config.channel_id in self.server.channels and self.config.channel_id:
            self.monitoring_channel: int = self.config.channel_id
        else:
            available_channels = list(self.server.channels.keys())
            self.monitoring_channel: int = random.choice(available_channels)
        self.persistance_prefix: str = self.config.persistance_prefix
        self.log_path: str = self.config.log_path
        self.persistance_path = self.config.persistance_path
        self.persistance_id: str = f"{self.persistance_prefix}_{self.archetype}" if self.persistance_prefix else ""
        self.plan: str = self.config.base_plan or "Responding to every message."
        self.sequential: bool = self.config.sequential_mode
        self.lock_response = False

        # creating necessary folders
        os.makedirs(self.persistance_path, exist_ok=True)
        os.makedirs(self.log_path, exist_ok=True)

        # archetype.yaml related attributes
        self.archetype_conf = DictToAttribute(
            **load_yaml('configs/archetypes.yaml')['agent_archetypes'][self.archetype])
        self.personnality_prompt: str = generate_agent_prompt(self.archetype, self.archetype_conf)
        self.name: str = self.archetype_conf.name

        # Logger
        self.logger = AgentLogger(self.persistance_id, self.config.log_path, self.config.log_level)
        self.logger.logger.info(f"Agent-Info: [key={self.name}] | Agent Configs loaded")

        # Agent Queues
        self.responses: Queue = Queue()
        self.processed_messages: Queue = Queue()
        self.event_queue: Queue[Event] = Queue()
        self.last_messages: deque = deque(maxlen=5)
        self.logger.logger.info(f"Agent-Info: [key={self.name}] | Queue created")

        # Agent State variable
        self.last_discussion_time = 0
        self.read_only: bool = False
        self._running: bool = True
        self.lock_queue: bool = False
        self.memory_count: int = 0
        self.logger.logger.info(f"Agent-Info: [key={self.name}] | State variable loaded")

        # Agent Modules
        self.responder = Responder(self.config.model)
        self.query_engine = QueryEngine(self.config.model)
        self.planner = Planner(self.config.model)
        self.contextualizer = Contextualizer(self.config.model)
        self.memory = db.Memories(collection_name=f'{self.persistance_id}_mem.pkl',
                                  base_folder=self.config.persistance_path)
        self.logger.logger.info(f"Agent-Info: [key={self.name}] | Module Loaded")

    # --- MISC ---

    def stop(self) -> None:
        """Stops agent modules at next iteration"""
        self._running = False

    async def add_event(self, event: Event) -> None:
        """
        Adds an event to the agent's queue if:
        - The agent is not the message author
        - The message is in the monitored channel
        - The event queue is not locked
        """

        if event.author_id != self.user_id and self.monitoring_channel == event.channel_id and not self.lock_queue:
            await self.event_queue.put(event)
            self.logger.logger.info(
                    f"Agent-Info: [key={self.name}] | Added event in event queue")


    # --- Module Helper ---

    def get_bot_context(self) -> str:
        """Returns real-time context: agent name, timestamp, and currently monitored channel."""
        return f"Your name is {self.name}. It is {datetime.now():%Y-%m-%d %H:%M:%S}. You are currently on discord reading the channel {self.server.get_channel(self.monitoring_channel)['name']}"

    async def get_channel_context(self, channel_id, bot_context) -> str:
        """
        Summarizes recent messages in a channel to provide a contextual backdrop. 
        Helps the agent retain short-term information without overloading the memory system.
        """
        msgs = [msg for msg in self.server.get_messages(channel_id).copy()]
        neutral_ctx = await self.contextualizer.summurize_transcript(msgs, bot_context)
        self.logger.log_event('neutral_ctxs', (msgs, bot_context), neutral_ctx)
        return neutral_ctx

    async def get_neutral_queries(self, channel_id) -> list[str]:
        """
        Generates neutral search queries (dialogue-based only, no plan/personality input) for memory retrieval.
        Used during planning to retrieve the most context-aware memories.
        """
        msgs = [msg for msg in self.server.get_messages(channel_id).copy()]
        context_queries = await self.query_engine.create_transcript_queries(msgs)
        self.logger.log_event('context_queries', msgs, context_queries)
        return context_queries

    async def get_memories(self, plan, context, messages) -> list[str]:
        """
        Queries memory using personality, plan, and current context to retrieve relevant reflections.
        Used during response generation to give agents consistent personalities.
        """
        queries = await self.query_engine.create_response_queries(plan, context, self.personnality_prompt, messages)
        self.logger.log_event('response_queries', (plan, context, self.personnality_prompt, messages), queries)
        memories = self.memory.query_multiple(queries)
        self.logger.log_event('memories', queries, memories)
        return memories

    async def get_response(self, plan, context, memories, messages, base_prompt) -> str:
        """
        Generates a user response. 
        Combines current context, short-term memory, long-term memory, and personality info.
        """
        response = await self.responder.make_response(plan, context, memories, messages, base_prompt, self.last_messages)
        self.logger.log_event('response', (plan, context, memories, messages, base_prompt), response)
        self.last_messages.append(response)
        return response

    async def get_reflection(self, messages, personality_prompt) -> str:
        """
        Generates a reflective summary based on recent messages.
        Used to store meaningful insights into long-term memory & allow agent to evolve.
        """
        reflection = await self.contextualizer.summurize_into_memory(messages, personality_prompt)
        self.logger.log_event('reflections', (messages, personality_prompt), reflection)
        return reflection

    async def get_plan(self, former_plan, context, unique_memories, channel_context, base_prompt) -> str:
        """
        Refines the agent's plan by incorporating new memories, context, and the personality prompt.
        """
        plan = await self.planner.make_plan(former_plan, context, unique_memories, channel_context, base_prompt)
        self.logger.log_event('plans', (former_plan, context, unique_memories, base_prompt), plan)
        return plan

    async def get_new_topic(self, plan, base_prompt) -> str:
        """
        Generates a spontaneous discussion topic.
        Used when idle to encourage diversity in conversation and memory creation.
        """

        return await self.responder.new_discussion(plan, base_prompt)

    # --- Routines

    # ------- Response Routine

    async def respond_routine(self) -> None:
        """
        Main routine that controls agent response behavior. Operates in both sequential and non-sequential modes.
        """

        while self._running:

            # If Empty Queue or Lock on response => Skip this iteration
            if self.event_queue.qsize() == 0 or self.lock_response:
                await asyncio.sleep(1)
                continue

            # In Sequential Mode: Always process the batch
            # Else:
            #   - Random Channel Switch May Occur followed by topic initiation
            #   - Random choice between processing batch, ignoring a message or reading the queue without responding.
            if self.sequential:
                await self._process_batch()
            else:
                # Channel Switch process the queue without responding
                # And a topic is initiated right away
                if random.random() < 0.05:
                    self.lock_queue = True
                    await self._read_only()
                    self.monitoring_channel = random.choice(
                        [channel_id for channel_id in self.server.channels.keys() if channel_id != self.monitoring_channel]
                    )

                    self.logger.logger.info(
                        f"Agent-Channel: [key={self.name}] | Switched to channel: {self.server.get_channel(self.monitoring_channel)}"
                    )

                    # New Topic Initiation
                    topic = await self.get_new_topic(self.plan, self.personnality_prompt)
                    if topic:
                        await self.responses.put((topic, self.monitoring_channel))
                        await self.processed_messages.put(f'[Me] {topic}')
                        self.logger.logger.info(f"Agent-Output: [key={self.name}] | Created new topic: {topic}")

                    self.lock_queue = False

                read_type = random.choices(['BATCH', 'IGNORE', 'ONLY_READ'], weights=[0.90, 0.025, 0.075], k=1)[0]

                if read_type == 'BATCH':
                    await self._process_batch()
                elif read_type == 'ONLY_READ':
                    await self._read_only()
                elif read_type == 'IGNORE':
                    await self._ignore()

                self.logger.logger.info(f"Agent-State: [key={self.name}] | Type of Read: {read_type}")
        

            await sleep(10)
            await sleep(random.uniform(0, self.config.max_random_response_delay))

    async def _process_messages(self, events) -> None:
        """
        Processes a list of messages by:
        - Formatting them for downstream modules
        - Retrieving context and memory
        - Generating and queuing a response (if applicable)
        - Adding messages to the processed message queue
        """

        formatted_messages = [self.server.format_message(event) for event in events]

        context = await self.get_channel_context(self.monitoring_channel, self.get_bot_context())
        memories = await self.get_memories(self.plan, context, formatted_messages)
        response = await self.get_response(self.plan, context, memories, formatted_messages, self.personnality_prompt)
        
        await self.responses.put((response, events[0].channel_id))

        for message in formatted_messages:
            await self.processed_messages.put(message)
            
        if response != "":
            await self.processed_messages.put(f'[Me] {response}')

    async def _process_batch(self) -> None:
        """
        Retrieves up to `noelem` messages from the event queue and processes them.
        If `noelem` is None, the entire queue is processed. It is used in both sequential & non-sequential mode.
        Indeed, if only putting one message at a time and only continuing when consuming a response, the agent is essentially synchone.
        """

        q_size = self.event_queue.qsize()

        batch = [
            await self.event_queue.get()
            for _ in range(q_size)
        ]
    
        if batch:
            self.logger.logger.info(
                f"Agent-Info: [key={self.name}] | Processed {q_size} elements from the event queue")
            await self._process_messages(batch)

    async def _read_only(self) -> None:
        """
        Reads all messages from the event queue and stores them as processed without responding.
        Useful for forming memories or during graceful transitions (e.g., channel switching) or simply make the agent more human.
        """
        current_size = self.event_queue.qsize()
        for _ in range(current_size):
            event = await self.event_queue.get()
            message = self.server.format_message(event)
            self.logger.logger.info(
                f"Agent-Info: [key={self.name}] | Processing message from {event.display_name} (read-only)")

            await self.processed_messages.put(message)

    async def _ignore(self) -> None:
        """
        Ignores one message from the event queue. 
        This message will not be reflected in context, memory, or response modules.
        """
        if not self.event_queue.empty():
            await self.event_queue.get()
            self.logger.logger.info(f"Agent-Info: [key={self.name}] | Ignoring Message in event queue")

    # ------- Plan Routine

    async def plan_routine(self) -> None:
        """
        Periodically evaluates and updates the agent's plan based on accumulated memory and context.

        Triggered every 6 memory events (using modulo check). Updates are then stored in memory for future reference.
        """
        await sleep(random.uniform(0, 180))
        while self._running and self.config.plans:
            await sleep(random.uniform(30, 120))
            try:
                self.logger.logger.info(f"Agent-Routine: [key={self.name}] | Started plan routine")
                if self.memory_count % 6 == 0 and self.memory_count != 0:
                    context = await self.get_channel_context(self.monitoring_channel, self.get_bot_context())
                    neutral_queries = await self.get_neutral_queries(self.monitoring_channel)
                    memories = self.memory.query_multiple(neutral_queries)
                    channel_context = await self.get_channel_context(self.monitoring_channel, self.get_bot_context())
                    updated_plan = await self.get_plan(self.plan, context, memories, channel_context,
                                                       self.personnality_prompt)
                    self.plan = updated_plan if updated_plan is not None else self.plan

                    if updated_plan:
                        self.memory_count += 1
                        self.memory.add_document(updated_plan, 'PLAN')
                        self.logger.logger.info(f"Agent-Routine: [key={self.name}] | Updated plan")
                else:
                    self.logger.logger.info(f"Agent-Routine: [key={self.name}] | Not enough memories to change plan")

            except Exception as e:
                self.logger.logger.error(f"Agent-Routine: [key={self.name}] | Error with planning routine: {e}")

    # ------- Memory Routine

    async def memory_routine(self) -> None:
        """
        Handles interaction with the processed_messages queue.

        If memory (reflection) is enabled, a reflection is triggered every 5 messages processed.

        Note: The _read_only() is given sense here as it will not trigger response, yet messages will be compacted into memories.
        """
        await sleep(random.uniform(0, 180))
        while self._running and self.config.memories:
            await sleep(random.uniform(20, 90))
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
                    self.logger.logger.info(
                        f"Agent-Routine: [key={self.name}] | Not enough message to process memories")
            except Exception as e:
                self.logger.logger.error(f"Agent-Routine: [key={self.name}] | Error with memory routine: {e}")
