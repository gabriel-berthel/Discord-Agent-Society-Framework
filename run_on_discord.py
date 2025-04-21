import argparse
import asyncio
import logging
import os
import sys

from dotenv import load_dotenv

import clients.discord_client as discord_client

logging.basicConfig(level=logging.INFO)

SERVER_CONFIG = 'configs/discord_server.yaml'

# Linux optimisations for asyncio.
if os.name != "nt":
    import uvloop

    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load .env and YAML configuration files for an agent.")
    parser.add_argument("env", type=str, help="Path to the agent config folder")

    args = parser.parse_args()

    if not os.path.isfile(args.env):
        print(f"Error: agent .env not found in '{args.env}'", file=sys.stderr)
        sys.exit(1)

    load_dotenv(args.env)

    discord_client.run(SERVER_CONFIG, os.getenv('ARCHETYPE'))
