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
    
    async def prompt(self, message, user_id, username):
        self.server.update_user(user_id, username)
        event = (1, user_id, username, message)
        self.agent.server.add_message(*event) 
        await self.agent.add_event(event)  
        message, _ = await self.agent.responses.get() 
        return message
    
    

async def exemple():
    server = DiscordServer(1, 'Benchmarking', 1)
    server.update_user(1, 'Joey')
    server.update_user(2, 'Interviewer')
    server.add_channel(1, 'General')
    
    joey = PromptClient('benchmark_config.yaml', 'trouble_maker', 'Rowan', 1, server)
    interviewer = PromptClient('benchmark_config.yaml', 'interviewer','Quinn', 2, server)
    
    await joey.start()
    await interviewer.start()

    joey_resp = await joey.prompt("Hi! How are you doing?", 2, 'Interviewer')
    print(joey_resp)
    while True:
        inter_resp = await interviewer.prompt(joey_resp, joey.id, joey.name)
        print(inter_resp)
        joey_resp = await joey.prompt(inter_resp, interviewer.id, interviewer.name)
        print("----")   
        print(joey_resp)

if __name__ == '__main__':
    asyncio.run(exemple())