from collections import deque

from models.event import Event


class DiscordServer:
    """
    Represents the Discord server in a virtualized form.

    This abstraction decouples the agent from direct client interactions,
    allowing the server to remain purely "virtual."

    It facilitates:
    - Logging the last 15 messages per channel, regardless of which channel the agent is actively monitoring.
    - Converting IDs to human-readable names.
    - Selecting appropriate channels when needed.

    By using this approach, no `DiscordApiWrapper` objects need to be passed to the agent.
    Instead, the agent interacts with a virtual `DiscordServer`, as exemplified in `prompt_client`.
    """

    def __init__(self, server_id, name):
        self.id = server_id
        self.name = name
        self.users: dict = {}
        self.channels: dict[int, dict] = {}

    def update_user(self, user_id, user_name) -> None:
        """Adds/Updates user dictionnary"""
        self.users[user_id] = user_name

    def add_channel(self, channel_id, channel_name) -> None:
        """Adds a channel. For each channel, the last 15 minutes are logged"""
        if channel_id not in self.channels:
            self.channels[channel_id] = {"name": channel_name, "messages": deque(maxlen=15), "last_id": None}

    def add_message(self, event: Event) -> None:
        """Add message to message circular queue"""

        if event.channel_id in self.channels:
            self.channels[event.channel_id]["messages"].append(self.format_message(event))
            self.channels[event.channel_id]["last_id"] = event.author_id

    def get_channel(self, channel_id) -> dict:
        """Returns channel dictionary"""
        return self.channels.get(channel_id, None)

    def get_messages(self, channel_id) -> list:
        """Returns the last 15 messages from channel"""
        return list(self.channels[channel_id]["messages"]) if channel_id in self.channels else []

    def fix_message(self, message) -> str:
        """Removes mentions from messages"""

        message = message.replace('\n', ' ')
        for user_id in self.users.keys():
            message = message.replace(f'<@{user_id}>', f'@{self.users[user_id]}')

        return message

    def format_message(self, event: Event) -> str:
        """Format messages in "User: Message" format"""
        return f"{event.display_name}: {self.fix_message(event.content)}"

    def __repr__(self) -> str:
        return f"DiscordServer({self.name}, {len(self.users)} users, {len(self.channels)} channels)"
