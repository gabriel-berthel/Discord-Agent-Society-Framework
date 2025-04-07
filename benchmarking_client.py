import asyncio
from agent import Agent
from modules.DiscordServer import DiscordServer

class BenchmarkingClient:
    def __init__(self, agent_conf, archetype):
        
        server = DiscordServer(1, 'Benchmarking')
        server.add_channel(1, 'General')
        
        server.update_user(1, 'Joey')
        server.update_user(2, 'Interviewer')
        
        self.agent = Agent(1, agent_conf, server, archetype, 'You must always respond to Interviewer.')
        
    async def start(self):
        print('Starting Argent Tasks')
        await asyncio.gather(
            self.agent.respond_routine(),
            self.agent.memory_routine(),
            self.agent.plan_routine(),
        )
    
    async def prompt(self, message):
        print(message)
        event = (1, 2, 'Interviewer', message)
        self.agent.server.add_message(*event)
        self.agent.event_queue.put(event)
        
        message, _ = await self.agent.responses.get() 
        print(message)
        return message
    
    async def run(self):
        await self.start()
        
    
async def main():
    agent_client = BenchmarkingClient('benchmark_config.yaml', 'trouble_maker')

    await agent_client.run()
    
    result = await agent_client.prompt("Hi! How are you doing?")
    print(result)

if __name__ == '__main__':
    asyncio.run(main()) 