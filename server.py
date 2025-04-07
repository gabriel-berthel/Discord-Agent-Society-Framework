import asyncio
import os
import discord_bot
import utils
import argparse
from dotenv import load_dotenv
import os
import sys

import logging
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("tqdm").setLevel(logging.ERROR) 

# Linux optimisations for asyncio.
if os.name != "nt":
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

def parse_arguments():
    """Parse arguments to ensure .env and .yaml files are provided"""

    parser = argparse.ArgumentParser(description="Load .env and YAML configuration files for an agent.")
    parser.add_argument("agent", type=str, help="Path to the agent config folder")

    args = parser.parse_args()

    # Check if .env file exists
    agent_conf = os.path.join(args.agent, 'agent.yaml')
    agent_env = os.path.join(args.agent, 'agent.env')

    if not os.path.isfile(agent_env):
        print(f"Error: agent.env not found in '{args.agent}'", file=sys.stderr)
        sys.exit(1)
    
    # Check if .yaml file exists
    if not os.path.isfile(agent_conf):
        print(f"Error: agent.conf not found in '{args.agent}'", file=sys.stderr)
        sys.exit(1)

    return args

if __name__ == "__main__":
    args = parse_arguments()
    load_dotenv(os.path.join(args.agent, 'agent.env'))
    agent_conf = utils.load_yaml(os.path.join(args.agent, 'agent.yaml'))
    discord_bot.run(agent_conf)
