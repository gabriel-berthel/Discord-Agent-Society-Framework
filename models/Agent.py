import logging
import os
import random
import time
from asyncio import Queue, sleep
from collections import deque
from datetime import datetime

import modules.agent_memories as db
from models.agent_logger import AgentLogger
from models.agent_state import AgentState
from models.discord_server import DiscordServer
from modules.agent_planner import Planner
from modules.agent_response_handler import Responder
from modules.agent_summuries import Contextualizer
from modules.query_engine import QueryEngine
from utils.agent_utils import *
from utils.base_prompt import generate_agent_prompt
from utils.file_utils import load_yaml

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
        self.event_queue: Queue = Queue()
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
        self.memory = db.Memories(collection_name=f'{self.persistance_id}_mem.pkl',
                                  base_folder=self.config.persistance_path)
        self.logger.logger.info(f"Agent-Info: [key={self.name}] | Module Loaded")

    # --- MISC ---

    def stop(self) -> None:
        """Stops agent modules at next iteration"""
        self._running = False

    async def add_event(self, event) -> None:
        """
        Adds an event to the agent's queue if:
        - The agent is not the message author
        - The message is in the monitored channel
        - The event queue is not locked
        """
        channel_id, author_id, _, _ = event

        # If queue is not locked, and agent is monitoring the channel
        if author_id != self.user_id and self.monitoring_channel == channel_id and not self.lock_queue:
            await self.event_queue.put(event)

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
        neutral_ctx = await self.contextualizer.neutral_context(msgs, bot_context)
        self.logger.log_event('neutral_ctxs', (msgs, bot_context), neutral_ctx)
        return neutral_ctx

    async def get_neutral_queries(self, channel_id) -> list[str]:
        """
        Generates neutral search queries (dialogue-based only, no plan/personality input) for memory retrieval.
        Used during planning to retrieve the most context-aware memories.
        """
        msgs = [msg for msg in self.server.get_messages(channel_id).copy()]
        context_queries = await self.query_engine.context_query(msgs)
        self.logger.log_event('context_queries', msgs, context_queries)
        return context_queries

    async def get_memories(self, plan, context, messages) -> list[str]:
        """
        Queries memory using personality, plan, and current context to retrieve relevant reflections.
        Used during response generation to give agents consistent personalities.
        """
        queries = await self.query_engine.response_queries(plan, context, self.personnality_prompt, messages)
        self.logger.log_event('response_queries', (plan, context, self.personnality_prompt, messages), queries)
        memories = self.memory.query_multiple(queries)
        self.logger.log_event('memories', queries, memories)
        return memories

    async def get_response(self, plan, context, memories, messages, base_prompt) -> str:
        """
        Generates a user response. 
        Combines current context, short-term memory, long-term memory, and personality info.
        """
        response = await self.responder.respond(plan, context, memories, messages, base_prompt, self.last_messages)
        self.logger.log_event('response', (plan, context, memories, messages, base_prompt), response)
        self.last_messages.append(response)
        return response

    async def get_reflection(self, messages, personality_prompt) -> str:
        """
        Generates a reflective summary based on recent messages.
        Used to store meaningful insights into long-term memory & allow agent to evolve.
        """
        reflection = await self.contextualizer.reflection(messages, personality_prompt)
        self.logger.log_event('reflections', (messages, personality_prompt), reflection)
        return reflection

    async def get_plan(self, former_plan, context, unique_memories, channel_context, base_prompt) -> str:
        """
        Refines the agent's plan by incorporating new memories, context, and the personality prompt.
        """
        plan = await self.planner.refine_plan(former_plan, context, unique_memories, channel_context, base_prompt)
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
        Main routine that controls agent response behavior.

        - **Sequential Mode**: Processes all queued events deterministically and continuously.
        ->> Channel switch must be handled externally.
        - **Non-Sequential Mode**: Adds random elements (e.g., ignore, read-only, random responses) for more lifelike behavior.
        ->> Also handles spontaneous topic initiation when idle, and supports switching between channels.
        """

        idle_threshold = 60 * 5
        last_active = time.time()

        # Sequential mode -> Agent always respond to everything in their queue ASAP
        if self.sequential:
            while self._running:
                if self.event_queue.qsize() > 0:
                    self.state = AgentState.PROCESSING
                    await self._process_batch()
                else:
                    self.state = AgentState.IDLE

                await sleep(self.config.response_delay)
                await sleep(random.uniform(0, self.config.max_random_response_delay))

        # Agent will behave more randomly
        # It's not always easy to keep track of what they're doing
        # But it's good as human are not predictable! sometimes they answer, sometime they read but forget, sometime they ignore...
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
                    self.monitoring_channel = random.choice([channel_id for channel_id in self.server.channels.keys() if
                                                             channel_id != self.monitoring_channel])
                    self.logger.logger.info(
                        f"Agent-Channel: [key={self.name}] | Switched to channel: {self.server.get_channel(self.monitoring_channel)}")
                    self.lock_queue = False

                # If no event in event queue, then agent will try to initiate topic.
                # self.read_only is not really set anywhere, but it's there to dynamically mute the agent if necessary.
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
                    substate = random.choices(['BATCH', 'IGNORE', 'ONLY_READ'],
                                              weights=[batch_weight, ignore_weight, only_read_weight], k=1)[0]
                    self.logger.logger.info(
                        f"Agent-Substate: [key={self.name}] | Substate: {substate}, w=[batch_weight={batch_weight}, ignore_weight={ignore_weight}, only_read_weight={only_read_weight}]")

                    if substate == 'BATCH':
                        # Will respond to a random number of elements in the queue.
                        # No need to check queue size as _process_batch takes the min.
                        noelem = int(random.uniform(1, 10))
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
                        self.logger.logger.info(f"Agent-Substate: [key={self.name}] | last_message_is_me=true")
                        # though there is 0.5 % chance the agent will send a new message anyway.
                        # so every cycle, if (ie 5 minutes) without any message after this agent sent one...
                        # agent may still try to bring a new topic!
                        # this bit of random is important as can diversify conversations
                        if not random.random() < 0.005:
                            self.logger.logger.info(
                                f"Agent-Substate: [key={self.name}] | Aborting topic initialisation")
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
            await sleep(self.config.response_delay)
            await sleep(random.uniform(0, self.config.max_random_response_delay))

    async def _process_messages(self, messages) -> None:
        """
        Processes a list of messages by:
        - Formatting them for downstream modules
        - Retrieving context and memory
        - Generating and queuing a response (if applicable)
        - Adding messages to the processed message queue
        """

        formatted_messages = [self.server.format_message(author_id, global_name, content) for
                              _, author_id, global_name, content in messages]

        context = await self.get_channel_context(self.monitoring_channel, self.get_bot_context())
        memories = await self.get_memories(self.plan, context, formatted_messages)
        response = await self.get_response(self.plan, context, memories, formatted_messages, self.personnality_prompt)

        if response:
            await self.responses.put((response, messages[0][0]))
        else:
            await self.responses.put(("", messages[0][0]))

        for message in formatted_messages:
            await self.processed_messages.put(message)

    async def _process_batch(self, noelem: int = None) -> None:
        """
        Retrieves up to `noelem` messages from the event queue and processes them.
        If `noelem` is None, the entire queue is processed. It is used in both sequential & non-sequential mode.
        Indeed, if only putting one message at a time and only continuing when consuming a response, the agent is essentially synchone.
        """

        q_size = self.event_queue.qsize()
        noelem = q_size if not noelem else min(q_size, noelem)

        batch = [
            await self.event_queue.get()
            for _ in range(noelem)
        ]

        if batch:
            self.logger.logger.info(
                f"Agent-Info: [key={self.name}] | Processed {noelem} elements in the {q_size} events from the event queue")
            await self._process_messages(batch)

    async def _read_only(self) -> None:
        """
        Reads all messages from the event queue and stores them as processed without responding.
        Useful for forming memories or during graceful transitions (e.g., channel switching) or simply make the agent more human.
        """
        current_size = self.event_queue.qsize()
        for _ in range(current_size):
            channel_id, author_id, global_name, content = await self.event_queue.get()
            message = self.server.format_message(author_id, global_name, content)
            self.logger.logger.info(
                f"Agent-Info: [key={self.name}] | Processing message from {global_name} (read-only)")
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
        while self._running and self.config.plans:
            await sleep(random.uniform(30, 120))
            try:
                self.logger.logger.info(f"Agent-Routine: [key={self.name}] | Started plan routine")
                if self.memory_count % 6 == 0:
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

            await sleep(30)

    # ------- Memory Routine

    async def memory_routine(self) -> None:
        """
        Handles interaction with the processed_messages queue.

        If memory (reflection) is enabled, a reflection is triggered every 5 messages processed.

        Note: The _read_only() is given sense here as it will not trigger response, yet messages will be compacted into memories.
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
                    self.logger.logger.info(
                        f"Agent-Routine: [key={self.name}] | Not enough message to process memories")
            except Exception as e:
                self.logger.logger.error(f"Agent-Routine: [key={self.name}] | Error with memory routine: {e}")

            await sleep(30)
