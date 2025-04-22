# How to run

## Generalities

1) For each agent, you must set up an application in the discord developer portal, invite it on your server, and enable
   all the intents. Make sure to use developer (setting on user UI) mode to grab server id.

2) Make sure to install ollama: https://ollama.com/

3) Pull the model specified in the config file (llama3:8b)

4) Make sure you have python 3.11 (3.12, 3.13 should work as well)

**Notes:** Agent not guaranteed to behave optimally with other models, without tweaking parameters (ie: temparature).

## Run Agent on discord

### Create new agent

Create **agent.env** file with the following keys:

```
TOKEN=<discord-api-token>
SERVER_ID=<discord-server-id>
ARCHETYPE=<archetype> (ie: troll)
```

**Each** agent should have their associated .env file!

### Install requirements

Run `pip install -r .\requirements.txt` to install dependencies. If you are on linux, also run `pip install uvloop`

### Run the agent

To run the agent on discord, start it using `python run_on_discord.py <ur .env file>`

Discord agent configuration is located in `configs/discord_server.yaml` and is **shared** across agents,
though feel free to modify *run_on_discord.py* if you want to have one config per discord agent.

**Note:** You can also define more archetypes in `archetype.yaml`

## Run the simulation outside discord:

Creating an instance of the Agent class, **any client** can be made, as all it takes is **reading** and **consuming**
from the queues. However, for **convenience**, the class `PromptClient` in `clients/prompt_client.py` provides a
*syncrhonous-ish* client. The method `run_simulation` creates the clients for all 5 archetypes and select the next one
to speak randomly.

If you wish to run the simulation that way, **outside discord**, you can simply run
`run_on_console.py <time in seconds>` and tweak `configs/console.yaml` if you wish to make any change to it.

