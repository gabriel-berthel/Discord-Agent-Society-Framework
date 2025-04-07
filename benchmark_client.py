import asyncio
import random
import os
from agent import Agent
from modules.DiscordServer import DiscordServer
import hikari

class BenchmarkingClient:
    def __init__(self, agent_conf, archetype):
        
        server = DiscordServer(1, 'Benchmarking')
        server.add_channel(1, 'General')
        
        server.update_user(1, 'Joey')
        server.update_user(2, 'Interviewer')
        
        self.agent = Agent(1, agent_conf, server, archetype, 'You must always respond to Interviewer.')
        
        asyncio.create_task(self.agent.respond_routine())
        asyncio.create_task(self.agent.memory_routine())
        asyncio.create_task(self.agent.plan_routine())
    
    async def prompt(self, message):
        event = (1, 2, 'Interviewer', message)
        self.agent.server.add_message(*event)
        self.agent.event_queue.put(event)
        
        message, _ = await self.agent.responses.get() 
        return message
    
agent = BenchmarkingClient('benchmark_config.yaml', 'trouble_maker')
