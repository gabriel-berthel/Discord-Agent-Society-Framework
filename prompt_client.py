import asyncio
import random
import time
from models.Agent import Agent
from models.DiscordServer import DiscordServer
import logging
from dotenv import load_dotenv

load_dotenv('agent.env')

logging.basicConfig(level=logging.WARNING)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("tqdm").setLevel(logging.ERROR) 
logging.getLogger("httpx").setLevel(logging.ERROR) 
logging.getLogger("modules.WebBrowser").setLevel(logging.ERROR) 

class PromptClient:
    def __init__(self, agent_conf, archetype, name, id, server):
        self.name = name
        self.id = id
        self.server = server
        self.agent = Agent(id, agent_conf, server, archetype)
        self.tasks = []
        self.server.update_user(id, self.agent.name)
        
    async def start(self):
        self.tasks = [
            asyncio.create_task(self.agent.respond_routine()),
            asyncio.create_task(self.agent.memory_routine()),
            asyncio.create_task(self.agent.plan_routine())
        ]
        
    async def stop(self):
        for task in self.tasks:
            task.cancel()
        for task in self.tasks:
            try:
                await task
                self.agent.stop()
            except asyncio.CancelledError:
                pass
    
    async def prompt(self, message, user_id, username, channel_id=1):
        self.server.update_user(user_id, username)
        event = (channel_id, user_id, username, message)
        self.agent.server.add_message(*event) 
        await self.agent.add_event(event)  
        message, _ = await self.agent.responses.get() 
        self.server.add_message(channel_id, user_id, username, message)
        return message
    
    @staticmethod
    def build_clients(config_file='benchmark_config.yaml'):
        server = DiscordServer(1, 'Benchmarking')
        server.add_channel(1, 'General')
        
        roles = [
            ('fact_checker', 'Caspian', 1),
            ('activist', 'Zora', 2),
            ('mediator', 'Quinn', 3),
            ('baseline', 'Neutri', 4),
            ('trouble_maker', 'Rowan', 5)
        ]
        
        clients = {
            role: PromptClient(config_file, role, name, client_id, server)
            for role, name, client_id in roles
        }

        return clients # dict -> archetype: client

    @staticmethod
    async def run_simulation(duration: float, print_replies, clients=None):
        clients = clients if clients else PromptClient.build_clients('benchmark_config.yaml')
        
        await asyncio.gather(*(client.start() for client in clients.values()))

        roles = list(clients.keys())
        start_time = time.time()
        historic = ['Hi']

        current_archetype = random.choice(roles)
        current_client = clients[current_archetype]
        message = "Hi"
        
        msg = f"[{current_client.name}] said {message}"
        historic.append(msg)
        
        if print_replies:
            print(msg)
        
        while time.time() - start_time < duration:
            archetype = [r for r in roles if r != current_archetype]
            next_archetype = random.choice(archetype)
            next_client = clients[next_archetype]

            response = await next_client.prompt(message, user_id=current_client.id, username=current_client.name)
            msg = f"[{next_client.name}] said {response}"
            historic.append(msg)
            
            if print_replies:
                print(msg)

            current_archetype = next_archetype
            current_client = next_client
            message = response

        return clients, historic


async def main():
    import os
    import pickle
    
    print_replies = True
    simulation_duration = 60 * 9 + 5
    clients, historic = await PromptClient.run_simulation(simulation_duration, print_replies)
    print(clients, historic)
    for archetype, client in clients.items():
        await client.stop()
        client.agent.save_logs()

    file_path = os.path.join("logs", f"qa_bench_histo.pkl")

    with open(file_path, "wb") as f:
        pickle.dump(historic, f)

    print(f"[LOG] Saved historic to {file_path}")   
    
    

if __name__ == "__main__":
    asyncio.run(main())
