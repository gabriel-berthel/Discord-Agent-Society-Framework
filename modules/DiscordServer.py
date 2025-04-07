from collections import deque
import hikari
import os

class DiscordServer:
    def __init__(self, server_id, name):
        self.id = server_id
        self.name = name
        self.users = {}
        self.channels = {}

    def update_user(self, user_id, user_name):
        self.users[user_id] = user_name

    def add_channel(self, channel_id, channel_name):
        if channel_id not in self.channels:
            self.channels[channel_id] = {"name": channel_name, "messages": deque(maxlen=25)}

    def add_message(self, channel_id, author_id, global_name, content):
        if channel_id in self.channels:
            self.channels[channel_id]["messages"].append(self.format_message(author_id, global_name, content))

    def get_channel(self, channel_id):
        return self.channels.get(channel_id, None)

    def get_messages(self, channel_id):
        return list(self.channels[channel_id]["messages"]) if channel_id in self.channels else []

    def __repr__(self):
        return f"DiscordServer({self.name}, {len(self.users)} users, {len(self.channels)} channels)"

    def fix_message(self, message):
        message = message.replace('\n', ' ')
        for user_id in self.users.keys():
            message = message.replace(f'<@{user_id}>', f'@{self.users[user_id]}')

        return message
    
    def format_message(self, author_id, global_name, content):
        name = "You" if author_id == int(os.getenv('USER_ID')) else global_name
        content = self.fix_message(content)

        return f"{name} sent: {content}"