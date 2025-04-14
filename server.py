import asyncio
import os
import discord_bot
import utils
import argparse
from dotenv import load_dotenv
import sys
import logging
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.WARNING)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("tqdm").setLevel(logging.ERROR) 
logging.getLogger("httpx").setLevel(logging.ERROR) 

# Linux optimisations for asyncio.
if os.name != "nt":
    import uvloop
    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

if __name__ == "__main__":
    print("hi")
    parser = argparse.ArgumentParser(description="Load .env and YAML configuration files for an agent.")
    parser.add_argument("env", type=str, help="Path to the agent config folder")
    parser.add_argument("config", type=str, help="Path to the agent config folder")
    parser.add_argument("archetype", type=str, help="Agent archetype")
    
    args = parser.parse_args()

    if not os.path.isfile(args.env):
        print(f"Error: agent .env not found in '{args.env}'", file=sys.stderr)
        sys.exit(1)
    
    if not os.path.isfile(args.config):
        print(f"Error: agent .yaml not found in '{args.config}'", file=sys.stderr)
        sys.exit(1)

    load_dotenv(args.env)

    discord_bot.run(args.config, args.archetype)
