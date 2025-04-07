# How to run

## Generalities

1) For each agent, you must set-ip an application on the discord developer portal.
**Important:** Make sure to enable ALL the intents.

2) Make sure to insall ollama: https://ollama.com/

3) If running on windows, you may have to install Microsoft C++ Build tool to run ChromaDB:
- Select "Desktop development with C++" workload.
- Select the "MSVC v14.x" version (e.g., MSVC v14.3x).
- Select Windows 10/11 SDK.
- https://visualstudio.microsoft.com/visual-cpp-build-tools/

## Agent Config 

Create **agent.env** file with the following keys:

```
TOKEN=<discord-api-token>
SERVER_ID=<discord-server-id>
USER_ID=<agent-account-user-id>
```

## Run Agent on discord

Run `pip install -r .\requirements.txt` to install dependencies.

To start the server, run `python server.py agent.env agent.yaml <archetype>` where archetype is either:
- baseline
- trouble_maker
- fact_checker
- activist
- moderator

**Note:** You can define more archetypes in `archetype.yaml`

## Run Agent on discord