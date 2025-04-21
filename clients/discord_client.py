import asyncio
import logging
import os

import hikari

import models.agent as ag
from models.discord_server import DiscordServer

logger = logging.getLogger(__name__)

agent = None
server = None
tasks = []


def run(agent_conf, archetype):
    async def message_handler():
        """
        Handles the message queue and sends messages to the appropriate Discord channel.
        Continuously checks the queue for messages and sends them to the channel.
        """
        logger.info("Message handler started")

        while True:
            message, channel_id = await agent.responses.get()
            logger.debug(f"Agent-Client: [key=Discord] | Dequeued message for channel {channel_id}: {message}")

            channel = (
                    bot.cache.get_guild_channel(channel_id)
                    or await bot.rest.fetch_channel(channel_id)
            )
            logger.debug(f"Agent-Client: [key=Discord] | Fetched channel object: {channel}")

            if message != "":
                await channel.send(message)
                logger.info(f"Agent-Client: [key=Discord] | Sent message to channel {channel_id}")
            else:
                logger.info(
                    f"Agent-Client: [key=Discord] | Empty message received and skipped for channel {channel_id}")

            agent.responses.task_done()

    bot = hikari.GatewayBot(
        intents=hikari.Intents.ALL,
        token=os.getenv("TOKEN")
    )

    @bot.listen(hikari.GuildMessageCreateEvent)
    async def on_message(event: hikari.GuildMessageCreateEvent):
        logger.debug(
            f"Agent-Client: [key=Discord] | Received message from {event.message.author.username}: {event.message.content}")
        await agent.add_event(
            (event.channel_id, event.message.author.id, event.message.author.display_name, event.message.content))
        agent.server.add_message(event.channel_id, event.message.author.id, event.message.author.display_name,
                                 event.message.content)
        logger.debug(f"Agent-Client: [key=Discord] | Message queued for processing and added to server representation")

    @bot.listen(hikari.MemberCreateEvent)
    async def on_member_create(event: hikari.MemberCreateEvent) -> None:
        user_id = event.user.id
        display_name = event.member.display_name if event.member else event.user.username
        agent.server.update_user(user_id, display_name)
        logger.info(f"Agent-Client: [key=Discord] | New member joined: {display_name} ({user_id})")

    @bot.listen(hikari.StoppingEvent)
    async def on_stopping(event: hikari.StoppingEvent) -> None:
        logger.info("Agent-Client: [key=Discord] | Bot is shutting down...")
        agent.stop()

        for task in tasks:
            task.cancel()
            logger.info(f"Agent-Client: [key=Discord] | Cancelled task: {task}")

        for task in tasks:
            try:
                await task
            except asyncio.CancelledError:
                logger.warning("Agent-Client: [key=Discord] | Unable to cancel task cleanly.")

    @bot.listen(hikari.GuildChannelCreateEvent)
    async def on_channel_create(event: hikari.GuildChannelCreateEvent) -> None:
        channel = event.channel
        if isinstance(channel, hikari.TextableChannel):
            server.add_channel(channel.id, channel.name)
            logger.info(f"Agent-Client: [key=Discord] | New channel created: {channel.name} ({channel.id})")

    @bot.listen(hikari.GuildChannelDeleteEvent)
    async def on_channel_delete(event: hikari.GuildChannelDeleteEvent) -> None:
        channel = event.channel
        if isinstance(channel, hikari.TextableChannel):
            server.remove_channel(channel.id)
            logger.info(f"Agent-Client: [key=Discord] | Channel deleted: {channel.name} ({channel.id})")

    @bot.listen(hikari.StartedEvent)
    async def on_started(event: hikari.StartedEvent):
        global agent, tasks, server
        logger.info("Agent-Client: [key=Discord] | Bot startup initiated")

        server_id = int(os.getenv("SERVER_ID"))
        uid = bot.get_me()
        guild = await bot.rest.fetch_guild(server_id)
        logger.info(f"CAgent-Client: [key=Discord] | onnected to server: {guild.name} ({server_id})")

        server = DiscordServer(server_id, guild.name)

        async for member in bot.rest.fetch_members(server_id):
            server.update_user(member.id, member.display_name)
            logger.debug(f"Agent-Client: [key=Discord] | Loaded member: {member.display_name} ({member.id})")

        for channel in await bot.rest.fetch_guild_channels(server_id):
            if isinstance(channel, hikari.TextableChannel):
                server.add_channel(channel.id, channel.name)
                logger.debug(f"Agent-Client: [key=Discord] | Loaded channel: {channel.name} ({channel.id})")

        agent = ag.Agent(uid, agent_conf, server, archetype)

        logger.info(f"Agent-Client: [key=Discord] | Loaded Server: {server}")
        logger.info(f"Agent-Client: [key=Discord] | Users: {server.users}")
        logger.info(f"Agent-Client: [key=Discord] | Channels: {server.channels}")

        tasks.append(asyncio.create_task(message_handler()))
        tasks.append(asyncio.create_task(agent.respond_routine()))
        tasks.append(asyncio.create_task(agent.memory_routine()))
        tasks.append(asyncio.create_task(agent.plan_routine()))

        logger.info("Agent-Client: [key=Discord] | All agent routines started. Bot is ready.")

    bot.run()
