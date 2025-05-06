# Summary

This project is designed to simulate a small society of AI agents — each with their own personality, memory, and
behavior. While it’s built to handle multi-agent conversations and dynamics, it also works just fine for running a
single agent in either a Discord server or directly in the terminal.

The codebase separates core functions like memory, planning, and response logic, making it easy to modify or extend.
Whether you're experimenting with agent interactions or just running a one-off bot, the system gives you a solid,
modular starting point.

# How to run

## Generalities

1) For each agent, you must set up an application in the discord developer portal, invite it on your server, and enable
   all the intents. Make sure to use developer (setting on user UI) mode to grab server id.

2) Make sure to install ollama: https://ollama.com/

3) Pull the model specified in the config file (llama3:8b)

4) Make sure you have python 3.11 (3.12, 3.13 should work as well)

**Notes:** Agent not guaranteed to behave optimally with other models, without tweaking parameters (ie: temparature).

## Install requirements

Run `python -m pip install -r requirements.txt` to install dependencies. If you are on linux, also run `pip install uvloop`

## Command-Line Interface (CLI) Usage Guide

**General usage:**  
`python hub.py <command> [options]`

---

### Commands & Arguments

#### 1. `discord`

Run a specified agent on a Discord server.  
**Note:** Provide environment variables either via a `.env` file [1] or individual arguments [2*]. All three parameters
below are required.

**Options:**

- `--env`         : *(string)* Path to a `.env` file containing environment variables. [1]
- `--token`       : *(string)* Discord bot token for authentication. [2a]
- `--server_id`   : *(string)* Discord server ID for bot deployment. [2b]
- `--archetype`   : *(string)* Agent archetype to initialize. [2c]

---

#### 2. `simulate`

Run a console-based simulation of available agents. Agents reply sequentially, with the next agent selected randomly.

**Options:**

- `--duration`    : *(int)* Duration of the simulation in seconds. Default: `3600`.
- `--verbose`     : *(flag)* Enable detailed logging during simulation.

---

#### 3. `prep_qa`

Prepare QA benchmark data by running a simulation and saving outputs to `output/qa_bench/*`.

**Options:**

- `--duration`    : *(int)* Time allocated for data preparation. Default: `3600`.
- `--verbose`     : *(flag)* Enable detailed output for preparation steps.

---

#### 4. `qa_bench`

Run QA benchmark tasks using outputs generated from `prep_qa`.  
**Requires:** Execution of `prep_qa` beforehand.

**Options:**

- `--verbose`     : *(flag)* Enable detailed output during benchmarking.

---

#### 5. `prob`

Probe an agent using a specified configuration and archetype. Opens a chat-like interface for interaction.

**Options:**

- `--config`      : *(string)* Path to the `.yaml` config file. **(Required)**
- `--archetype`   : *(string)* Name of the agent archetype. **(Required)**

#### 5. `promptbench`

Run the promptbench benchmarking tasks.

---

*Notes:* Use `--help` with any subcommand for detailed usage, e.g., `python hub.py discord --help`

## Configuration Files

**General configuration file**:

- `archetypes.yaml`: define the archetypes here
- `ollama_options.py`: base-model tweaking here (per module)

**Agent Yaml Configuration Files are located in `configs\clients\`**:

- (1) `discord.yaml`: Used running the simulation on discord
- (2) `simulate.yaml`: Used running the simulation in the console
- (3) `prep_qa.yaml`: Used to prepare Quantative Benchmark data
- (4) `qa_bench.yaml`: Used Compute Quantitative Benchmark
- (5) `promptbench.yaml`: parameters for the promptbench benchmarking

## Run the simulation

### On the console

- Make sure the config file sets the agents in sequential mode
- Run  `python hub.py simulate --duration <duration in seconds> --verbose`

### On a discord server

Make sure to:

- Create (1) application per agent
- Enable ALL intents on the application portal
- Invite the application inside the server

For each agent create a .env with the following keys:

- `TOKEN`: discord application token
- `SERVER_ID`: server ID the should operate in
- `ARCHETYPE`: archetype of the agent (defined in `configs/archetypes.yaml`)

Then run `python hub.py discord --env <agent_env_file>`

Alternatively, you can run `python hub.py discord --token <token> --server_id <id> --archetype <archetype>`

## Run benchmarks

TODO: Write this part

## Some technical details

### Modules

- `agent_memories.py`: Handles Vector Database for Memory Retrival
- `agent_planner.py`: Handles Agent Planning
- `agent_response_handler.py`: Handles Agent Responses
- `agent_summuries.py`: create contextual summaries & agent memories
- `query_engine.py`: creates queries used for memory retrival

### Important Models

#### `agent.py` — Agent Lifecycle and Communication Orchestrator

This object encapsulates the internal state and behavior of an AI Agent. It handles asynchronous communication with
external clients via queues and coordinates with internal modules through structured routines.

##### Asynchronous I/O Communication

- **`responses`** — Outbound messages from the agent to the client.
    - Emission is guaranteed if `sequential` mode is enabled in the configuration.
    - Must be consumed by an external asynchronous client.

- **`event_queue`** — Inbound messages from the client to the agent (primary input channel).
    - Messages are inserted by an external client.
    - Monitored and consumed by the Response Routine.
    - May be processed in "read-only", "ignore", or "batch" modes.

- **`_processed_message_queue`** — Internal staging for memory formation.
    - Populated when events are processed in "read-only" or "batch" modes.
    - Consumed by the Memory Routine to generate and store compacted memories.

##### Module Communication

- **Planning Routine**
    - Triggered every 5 memories (if enabled).
    - Handles high-level planning tasks such as goal setting or dialogue structuring.

Pseudocode:

```text
While Plan Routine Is Running:
If there are more than 5 new memories:
  - Fetch last 15 messages from the relevant Discord channel
  - Generate a Summary using Agent Summaries module
  - Generate Neutral Queries from the Summary
  - Retrieve related Memories using the Query Engine
  - Fetch channel metadata
  - Execute Plan method in Agent Planner
  - If the Plan is updated:
      - Store the Plan in the Vector Memory Database as a new memory
```

- **Memory Routine**  
  **Input**: `_processed_message_queue`  
  **Output**: Writes to the memory module.
    - Invoked every 5 processed messages (if enabled).
    - Updates long-term or contextual memory from compacted experiences.

Pseudocode:

```text
While Memory Routine is running:
If processedMessagesQueue has 5 or more messages:
  - Retrieve 5 messages
  - Generate a Reflection using Agent Summaries module
  - Store the Reflection in the Vector Memory Database
  - Increment the reflection counter
```

- **Response Routine**  
  **Input**: `event_queue`  
  **Outputs**: `_processed_message_queue`, `responses`
    - Manages message interpretation and determines reply strategy.
    - Supports both deterministic and probabilistic behavior modes:

    - **Sequential Mode (`sequential = True`)**
        - Messages are processed and responded to immediately (synchronous-like behavior).

    - **Asynchronous Mode (`sequential = False`)**
        - Introduces dynamic, realistic behavior:
            - Random channel switching
            - Selective message ignoring
            - “Read-only” mode for silent memory formation
            - Batched processing of event queue
            - Initiates spontaneous messages when switching channel

Pseudocode:

```text
1. Queue & Lock Check
- If the event queue is empty OR response lock is active:
  -> Wait 1 second
  -> Skip this iteration

2. Sequential Mode (self.sequential == True)
- Always:
  -> Process the entire event queue (batch mode)

3. Non-Sequential Mode

- 5% chance to trigger a channel switch:
  -> Lock the event queue
  -> Read messages without responding
  -> Randomly switch to a different monitoring channel
  -> Generate a new topic using the current plan and personality prompt
     If a message is generated:
       -> Add it to the response queue
       -> Add it to the processed message queue
  -> Unlock the event queue

- Randomly select behavior (weighted):
  - 90% chance — Respond to event queue:
    -> Generate a contextual summary from the last 15 messages
    -> Retrieve relevant memories using:
       - Contextual summary
       - Current plan
       - Event queue messages
       -> Query Retriever generates queries
       -> Vector database returns matching memories
    -> Create a response using:
       - Plan
       - Context
       - Retrieved memories
       - Event queue messages
       - Personality prompt
    -> Add the response to the response queue
    -> Move the processed event message to the processed message queue

  - 7.5% chance — Read Only:
    -> Move the message to the processed message queue without responding

  - 2.5% chance — Ignore:
    -> Discard one event from the event queue with no further action
```

- **Module Interactions**
    - **DiscordServer**: Enables contextual awareness through channel switching and summarization of recent messages (up
      to 15).
    - **Memory Module**: In-memory by default with persistence via pickling. Easily swappable with other backends
      implementing the same interface.

---

#### `discord_server.py` — Virtual Representation of the Discord Server

Acts as an abstracted, client-managed representation of a Discord-like environment, enabling decoupling between the
agent logic and the communication platform.

Key Points:

- Must be instantiated and managed externally (by the client).
- Enables the agent to:
    - Switch channels based on context or events.
    - Retrieve the last *n* messages from a specific channel.
- Enable ping translation (`@<DiscordId>` to display name).
- Designed to generalize across any channel-based communication backend.

### Clients

Discord is merely the front end of the project,
though the nature of the code make it functionally separate from the discord layer.

#### `discord_client.py` — Agent Deployment in a Discord Environment

- Manages the lifecycle of an agent within a live Discord server.
- Interfaces with `discord_server.py` to provide the agent with channel-based context and message events.
- Responsible for:
    - Connecting to Discord APIs.
    - Routing messages from channels into the agent’s event queue.
    - Delivering agent responses back into the appropriate Discord channels.

#### `prompt_client.py` — Console-Based Agent Runner

- Provides a CLI-based interface for interacting with agents.
- Primary use cases:
    - **Benchmarking**: Evaluate agent behavior and performance in a controlled terminal setting.
    - **Chat-like Prompting**: Manually probe or interact with the agent for Q&A, debugging, messing around or research.
    - **Run console simulation**: Instantiate all archetype and randomly selects next responder each turn.

- Ideal for lightweight experimentation without requiring a full Discord environment.
- Assumes the config sets the bot to be sequential

**Note:**  
Delays or lag may occur when all agents plan or write memories simultaneously.  
Since they operate within the same execution thread, wait calls can accumulate, leading to performance bottlenecks.  
Although the prompt client is asynchronous, it behaves in a somewhat synchronous manner under these conditions.

