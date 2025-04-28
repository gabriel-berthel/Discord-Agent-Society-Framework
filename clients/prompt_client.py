import asyncio
import logging
import os
import pickle
import random
import shutil
import time
import asyncio

from dotenv import load_dotenv

from models.agent import Agent
from models.event import Event
from models.discord_server import DiscordServer

logger = logging.getLogger(__name__)

class PromptClient:
    def __init__(self, agent_conf, archetype, user_id, server):
        self.server: DiscordServer = server
        self.agent = Agent(user_id, agent_conf, server, archetype)
        self.tasks = []
        self.name = self.agent.name
        self.server.update_user(user_id, self.agent.name)

        logger.info(
            f"Agent-Client: [key=PromptClient] | [{self.name}] Initialized with archetype '{archetype}', ID: {self.server.id}")

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

    async def prompt(self, message, user_id, username, channel_id=1):
        logger.info(f"Agent-Client: [key=PromptClient] | [{self.name}] Prompting with message: '{message}'")

        prompted = Event(
            channel_id=channel_id,
            author_id=user_id,
            display_name=username,
            content=message,
        )

        await self.agent.add_event(prompted)
        self.server.add_message(prompted)

        message, _ = await self.agent.responses.get()
        logger.info(f"Agent-Client: [key=PromptClient] | [{self.name}] Received response: '{message}'")

        self._add_message_from_agent_to_server(content=message)

        logger.info(f"Agent-Client: [key=PromptClient] | [{self.name}] Final response: '{message}'")
        return message

    def _add_message_from_agent_to_server(self,content):

        event = Event(
            channel_id=self.agent.monitoring_channel,
            author_id=self.agent.user_id,
            display_name=self.agent.name,
            content=content,
        )

        self.server.add_message(event)

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
            role: PromptClient(config_file, role, client_id, server)
            for role, name, client_id in roles
        }

        logger.info("Agent-Client: [key=PromptClient] | Prompt clients built successfully.")
        return clients

    @staticmethod
    async def run_simulation(duration: float, verbose: bool, config_file:str, initial_message="Hi! What's up gamers"):
        """
        Helper method to run a simulation a quasi-synchronous way.

        Indeed, agents are put in sequential-mode. Though, the next speaker is determined randomly.

        Client response routine is "locked" until their time to respond comes. This ensures the entire batch
        is processed all at once. ie: if 10 messages were sent before a client is randomly selected, the agent
        tied to the client will respond to the 10 messages.

        """

        logger.info(f"Agent-Client: [key=PromptClient] | Running simulation for {duration} seconds.")
        transcript = []

        # fetching clients if none are given
        clients = PromptClient.build_clients(config_file)

        # Lock response routine so the event queue is only processed when desired
        for client in clients.values():
            client.agent.monitoring_channe = 1
            client.agent.lock_response = True

        # grabbing all the roles
        roles = [role for role, client in clients.items()]

        # starting clients
        await asyncio.gather(*(client.start() for client in clients.values()))
        logger.info("Agent-Client: [key=PromptClient] | All clients started.")

        start_time = time.time()

        # Sending the first message
        current_archetype = random.choice(roles)
        current_client = clients[current_archetype]
        message = f"[{current_client.name}] {initial_message}"
        verbose and print(message)
        transcript.append(message)

        prompt = initial_message
        while time.time() - start_time < duration:
            logger.info(
                f"Agent-Client: [key=PromptClient] | [{current_client.name}] Broadcasting messages to next agent.")

            event = Event(
                channel_id=1,
                author_id=current_client.agent.user_id,
                display_name=current_client.agent.name,
                content=prompt
            )

            # Adding event, to agents event queue
            # Current agent filtering isn't done as the event queue filters it.
            for client in clients.values():
                await client.agent.add_event(event)

            # Choosing next Agent to respond
            next_archetype = random.choice([r for r in roles if r != current_archetype])
            next_client: PromptClient = clients[next_archetype]

            # Responding to every event before
            next_client.agent.lock_response = False
            logger.info(
                f"Agent-Client: [key=PromptClient] | [{next_client.name}] Responding to message from previous agent.")
            response, channel_id = await next_client.agent.responses.get()
            next_client.agent.lock_response = True

            # Just logging and storing messages.
            message = f"[{next_client.name}] {response}"
            verbose and print(message)
            transcript.append(message)

            # Preparing next irritation
            current_archetype, current_client = (next_archetype, next_client)
            prompt = response

        logger.info("Agent-Client: [key=PromptClient] | Simulation completed.")
        return clients, transcript

    @staticmethod
    async def prepare_qa_bench(duration, print_replies):
        # Set up the output directory
        shutil.rmtree('output/qa_bench', ignore_errors=True)
        os.makedirs('output/qa_bench')
        os.makedirs('output/qa_bench/logs')
        os.makedirs('output/qa_bench/memories')

        logger.info('Running simulation')
        clients, historic = await PromptClient.run_simulation(
            duration, print_replies, config_file='configs/clients/prep_qa.yaml'
        )

        logger.info('Saving logs in output/qa_bench/logs')
        for archetype, client in clients.items():
            await client.stop()
            client.agent.logger.save_logs()

        logger.info('Saving historic in output/qa_bench/')
        with open("output/qa_bench/qa_bench_histo.pkl", "wb") as f:
            pickle.dump(historic, f)
        with open("output/qa_bench/qa_bench_histo.txt", "w") as f:
            for line in historic:
                f.writelines(f'{line}\n')
