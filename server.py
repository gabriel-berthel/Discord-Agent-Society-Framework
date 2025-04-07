import asyncio
import os
import discord_bot
import utils
import argparse
from dotenv import load_dotenv
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load .env and YAML configuration files for an agent.")
    parser.add_argument("env", type=str, help="Path to the agent config folder")
    parser.add_argument("config", type=str, help="Path to the agent config folder")
    
    args = parser.parse_args()

    if not os.path.isfile(args.env):
        print(f"Error: agent .env not found in '{args.env}'", file=sys.stderr)
        sys.exit(1)
    
    if not os.path.isfile(args.config):
        print(f"Error: agent .yaml not found in '{args.config}'", file=sys.stderr)
        sys.exit(1)

    load_dotenv(args.env)

    agent_conf = utils.load_yaml(args.config)

    discord_bot.run(agent_conf)
