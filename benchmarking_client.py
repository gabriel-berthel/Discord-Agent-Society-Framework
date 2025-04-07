import asyncio
from agent import Agent
from modules.DiscordServer import DiscordServer
from dotenv import load_dotenv

class BenchmarkingClient:
    def __init__(self, agent_conf, archetype):
        
        server = DiscordServer(1, 'Benchmarking', 1)
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
        event = (1, 2, 'Interviewer', message)
        self.agent.server.add_message(*event)
        await self.agent.event_queue.put(event)
        
        message, _ = await self.agent.responses.get() 
        return message
    
    async def run(self):
        await self.start()
        
async def main():
    agent_client = BenchmarkingClient('benchmark_config.yaml', 'trouble_maker')

    agent_client.run()
    print("hii")
    result = await agent_client.prompt("Hi! How are you doing?")
    print(result)

if __name__ == '__main__':
    asyncio.run(main()) 