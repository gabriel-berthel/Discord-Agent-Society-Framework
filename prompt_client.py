import asyncio
from agent import Agent
from modules.DiscordServer import DiscordServer
from dotenv import load_dotenv

class PromptClient:
    def __init__(self, agent_conf, archetype, name, id, server):
        self.name = name
        self.id = id
        self.server = server
        self.agent = Agent(id, agent_conf, server, archetype, 'You must always respond to Interviewer.')
        self.server.update_user(id, self.agent.name)
        
    async def start(self):
        asyncio.create_task(self.agent.respond_routine())
        asyncio.create_task(self.agent.memory_routine())
        asyncio.create_task(self.agent.plan_routine())
    
    async def prompt(self, message, user_id, username, channel_id=1):
        self.server.update_user(user_id, username)
        event = (channel_id, user_id, username, message)
        self.agent.server.add_message(*event) 
        await self.agent.add_event(event)  
        message, _ = await self.agent.responses.get() 
        return message
    
    def build_clients(config_file='benchmark_config.yaml'):
        server = DiscordServer(1, 'Benchmarking')
        server.add_channel(1, 'General')
        
        roles = [
            ('fact_checker', 'Caspian', 1),
            ('activist', 'Zora', 2),
            ('interviewer', 'Quinn', 3),
            ('baseline', 'Neutri', 4),
            ('trouble_maker', 'Rowan', 5)
        ]

        clients = [
            PromptClient(config_file, role, name, client_id, server)
            for role, name, client_id in roles
        ]

        return tuple(clients)

