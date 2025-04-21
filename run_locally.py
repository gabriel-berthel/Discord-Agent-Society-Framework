import asyncio
import os
import sys
import logging
from clients.prompt_client import PromptClient

logging.getLogger("clients.prompt_client").setLevel(logging.WARNING)

SERVER_CONFIG = 'configs/local.yaml'

# Linux optimisations for asyncio.
if os.name != "nt":
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

CONFIG = 'configs/local.yaml'

async def run_locally(runtime: float):
    await PromptClient.run_simulation(runtime, True, CONFIG, 'Hi you all :333')
    
if __name__ == "__main__":
    runtime = float(sys.argv[1])
    asyncio.run(run_locally(runtime))
