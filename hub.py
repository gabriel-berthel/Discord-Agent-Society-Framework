import argparse
from dataclasses import dataclass
from dotenv import load_dotenv
import asyncio
import subprocess
import os
import logging


from clients.prompt_client import PromptClient
import clients.discord_client as discord_client
from models.discord_server import DiscordServer
from benchmark.qa_benchmark import load_qa_bench_data, run_benchmarks

if os.name != "nt":
    import uvloop

    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())

SERVER_CONFIG = 'configs/clients/discord.yaml'
CONSOLE_SIMULATION_CONFIG = 'configs/clients/simulate.yaml'

# ---------- Configs ----------
@dataclass
class DiscordConfig:
    env_path: str | None = None
    token: str | None = None
    server_id: str | None = None
    archetype: str | None = None

@dataclass
class SimConfig:
    duration: int
    verbose: bool

@dataclass
class BenchPrepConfig:
    duration: int
    verbose: bool

@dataclass
class RunBenchConfig:
    verbose: bool

@dataclass
class ProbingConfig:
    config: str
    archetype: str


# ---------- Command Handlers ----------
def run_discord_bot(config: DiscordConfig):
    if config.env_path:
        load_dotenv(config.env_path)
        print(f"Loaded environment from {config.env_path}")

    token = config.token or os.getenv("TOKEN")
    server_id = config.server_id or os.getenv("SERVER_ID")
    archetype = config.archetype or os.getenv("ARCHETYPE", "nerd")

    if not token or not server_id:
        raise ValueError("Missing required Discord credentials (token/server_id).")

    print(f"Running Discord bot with archetype '{archetype}' on server {server_id}...")

    os.environ["TOKEN"] = token
    os.environ["SERVER_ID"] = server_id
    os.environ["ARCHETYPE"] = archetype

    discord_client.run(SERVER_CONFIG)

async def run_simulation(config: SimConfig):
    if config.verbose:
        print(f"Running simulation for {config.duration} seconds...")
    await PromptClient.run_simulation(config.duration, True, CONSOLE_SIMULATION_CONFIG, 'Hi!')

async def prepare_qa_bench(config: BenchPrepConfig):
    if config.verbose:
        print(f"Preparing benchmark data for {config.duration} seconds...")
    await PromptClient.prepare_qa_bench(config.duration, config.verbose)

async def benchmark_qa_bench(config: RunBenchConfig):
    print("Downloading en_core_web_sm")
    subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])

    logs = load_qa_bench_data()
    if config.verbose:
        print("Loaded logs, running benchmarks...")

    await run_benchmarks(logs)

async def probe(config: ProbingConfig):
    server = DiscordServer(1, 'Probing')
    server.add_channel(1, 'Chat')
    client = PromptClient(config.config, config.archetype, -1, server)
    client.agent.sequential = True

    await client.start()

    while True:
        q = input("Enter your question (type 'exit' to stop): ")

        if q.lower() == 'exit':
            print("Ending the Q&A session.")
            break

        a = await client.prompt(q, 128, 'Moderator', 1)
        print('Agent:', a)
    await client.stop()

# ---------- Main CLI ----------
def main():
    parser = argparse.ArgumentParser(description="AgentHub CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Discord bot
    p_discord = subparsers.add_parser("discord", help="Run the Discord agent")
    p_discord.add_argument("--env", type=str, help="Path to .env file")
    p_discord.add_argument("--token", type=str, help="Discord bot token")
    p_discord.add_argument("--server_id", type=str, help="Discord server ID")
    p_discord.add_argument("--archetype", type=str, help="Agent archetype")

    # Simulation
    p_sim = subparsers.add_parser("simulate", help="Run console simulation")
    p_sim.add_argument("--duration", type=int, default=3600)

    # QA Benchmark Prep
    p_prep = subparsers.add_parser("prep_qa", help="Prepare QA benchmark data")
    p_prep.add_argument("--duration", type=int, default=3600)
    p_prep.add_argument("--verbose", action="store_true")

    # Start QA benchmark
    p_bench = subparsers.add_parser("qa_bench", help="Run QA benchmark tasks. Requires u ran prep_qa first")
    p_bench.add_argument("--verbose", action="store_true")

    # Probing
    p_prob = subparsers.add_parser("prob", help="Probe an agent")
    p_prob.add_argument("--config", type=str, required=True, help="Path to the agent .yaml to use")
    p_prob.add_argument("--archetype", type=str, required=True, help="Archetype to use for probing")

    args = parser.parse_args()

    # Dispatch
    match args.command:
        case "discord":
            # Either discord .env or provide everything right away
            if args.env:
                if args.token or args.server_id or args.archetype:
                    parser.error("Cannot use '--env' with '--token', '--server_id', or '--archetype'.")
            else:
                if not (args.token and args.server_id and args.archetype):
                    parser.error("Must provide '--token', '--server_id', and '--archetype' if '--env' is not used.")

            run_discord_bot(DiscordConfig(
                env_path=args.env,
                token=args.token,
                server_id=args.server_id,
                archetype=args.archetype
            ))
        case "simulate":
            asyncio.run(run_simulation(SimConfig(args.duration, args.verbose)))
        case "prep_qa":
            print(f'Starting prep_qa with --duration={args.duration} and --verbose={args.verbose}')
            asyncio.run(prepare_qa_bench(BenchPrepConfig(args.duration, args.verbose)))
        case "qa_bench":
            print('Please make sure you ran prep_qa first!')
            asyncio.run(benchmark_qa_bench(RunBenchConfig(args.verbose)))
        case "prob":
            asyncio.run(probe(ProbingConfig(args.config, args.archetype)))


if __name__ == "__main__":
    main()
