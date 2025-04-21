import asyncio
import random
import time
from models.agent import Agent
from models.discord_server import DiscordServer
import logging
from dotenv import load_dotenv

load_dotenv('agent.env')

logger = logging.getLogger(__name__)

class PromptClient:
    def __init__(self, agent_conf, archetype, name, id, server):
        self.name = name
        self.id = id
        self.server: DiscordServer = server
        self.agent = Agent(id, agent_conf, server, archetype)
        self.tasks = []
        self.server.update_user(id, self.agent.name)
        logger.info(f"Agent-Client: [key=PromptClient] | [{self.name}] Initialized with archetype '{archetype}', ID: {id}")

    async def start(self):
        logger.info(f"Agent-Client: [key=PromptClient] | [{self.name}] Starting agent routines.")
        self.tasks = [
            asyncio.create_task(self.agent.respond_routine()),
            asyncio.create_task(self.agent.memory_routine()),
            asyncio.create_task(self.agent.plan_routine())
        ]
        logger.info(f"Agent-Client: [key=PromptClient] | [{self.name}] Agent routines started.")

    async def stop(self):
        logger.info(f"Agent-Client: [key=PromptClient] | [{self.name}] Stopping agent and cancelling tasks.")
        self.agent.stop()
        for task in self.tasks:
            task.cancel()
        for task in self.tasks:
            try:
                await task
                logger.info(f"Agent-Client: [key=PromptClient] | [{self.name}] Task completed cleanly.")
            except asyncio.CancelledError:
                logger.warning(f"Agent-Client: [key=PromptClient] | [{self.name}] Task cancellation was not clean.")

    async def prompt(self, message, user_id, username, channel_id=1, log_file=None):
        logger.info(f"Agent-Client: [key=PromptClient] | [{self.name}] Prompting with message: '{message}'")
        self.server.update_user(user_id, username)
        event = (channel_id, user_id, username, message)
        self.server.add_message(channel_id, user_id, username, message)
        
        if log_file:
            with open(log_file, 'a') as file:
                print('q:', message)
                file.write(f"q: [{message}]\n")
        
        await self.agent.add_event(event)

        attempt = 0
        message, _ = await self.agent.responses.get()
        logger.info(f"Agent-Client: [key=PromptClient] | [{self.name}] Received response: '{message}'")

        while attempt < 5 and message == "":
            logger.info(f"Agent-Client: [key=PromptClient] | [{self.name}] Empty response attempt {attempt + 1}, retrying...")
            message, _ = await self.agent.responses.get() 
            attempt += 1

        if log_file:
            with open(log_file, 'a') as file:
                print('a:', message)
                file.write(f"a: [{message}]\n")
        
        self.server.add_message(channel_id, user_id, username, message)
        logger.info(f"Agent-Client: [key=PromptClient] | [{self.name}] Final response: '{message}'")
        return message

    async def multi_prompt(self, events):
        logger.info(f"Agent-Client: [key=PromptClient] | [{self.name}] Executing multi_prompt with {len(events)} events.")
        for message, user_id, username, channel_id in events:
            self.server.update_user(user_id, username)
            event = (channel_id, user_id, username, message)
            await self.agent.add_event(event)
            logger.info(f"Agent-Client: [key=PromptClient] | [{self.name}] Event added: [{username}] -> '{message}'")
            
        message, _ = await self.agent.responses.get() 
        self.server.add_message(channel_id, user_id, username, message)
        logger.info(f"Agent-Client: [key=PromptClient] | [{self.name}] Multi-prompt response: '{message}'")
        return message

    @staticmethod
    def build_clients(config_file='benchmark_config.yaml'):
        logger.info("Agent-Client: [key=PromptClient] | Building prompt clients.")
        server = DiscordServer(1, 'Benchmarking')
        server.add_channel(1, 'General')
        
        roles = [
            ('debunker', 'Caspian', 1),
            ('nerd', 'Zora', 2),
            ('peacekeeper', 'Quinn', 3),
            ('chameleon', 'Neutri', 4),
            ('troll', 'Rowan', 5)
        ]
        
        clients = {
            role: PromptClient(config_file, role, name, client_id, server)
            for role, name, client_id in roles
        }

        logger.info("Agent-Client: [key=PromptClient] | Prompt clients built successfully.")
        return clients

    @staticmethod
    async def run_simulation(duration: float, print_replies, config_file, initial_message="Hi! What's up gamers", clients=None):
        logger.info(f"Agent-Client: [key=PromptClient] | Running simulation for {duration} seconds.")
        clients = clients if clients else PromptClient.build_clients(config_file)

        for client in clients.values():
            await client.start()
        
        logger.info("Agent-Client: [key=PromptClient] | All clients started.")
        roles = list(clients.keys())
        start_time = time.time()
        historic = [initial_message]

        current_archetype = random.choice(roles)
        current_client = clients[current_archetype]
        message = initial_message

        msg = f"[{current_client.name}] {message}"
        if print_replies:
            print(msg)

        agent_histories = {role: [] for role in roles}

        while time.time() - start_time < duration:
            logger.info(f"Agent-Client: [key=PromptClient] | [{current_client.name}] Broadcasting message to next agent.")

            for role in roles:
                if role != current_archetype:
                    agent_histories[role].append((message, current_client.id, current_client.name, 1))

            next_archetype = random.choice([r for r in roles if r != current_archetype])
            next_client = clients[next_archetype]

            context_events = agent_histories[next_archetype]
            logger.info(f"Agent-Client: [key=PromptClient] | [{next_client.name}] Responding to message from previous agent.")

            response = await next_client.multi_prompt(context_events)

            msg = f"[{next_client.name}] {response}"
            if print_replies:
                print(msg)

            historic.append(msg)
            agent_histories[next_archetype] = []
            current_archetype = next_archetype
            current_client = next_client
            message = response

        logger.info("Agent-Client: [key=PromptClient] | Simulation completed.")
        return clients, historic
