import hikari
import os
import random
import asyncio
import agent as ag
from entities.DiscordServer import DiscordServer

agent = None

def run(agent_conf):
    async def initialize_known_users(agent):
        for guild in bot.get_guilds():
            async for member in guild.fetch_members():
                agent.known_users[member.id] = member.global_name

    async def message_handler():
        """Handles message queue and repond in the appropriate channel."""
        
        while True:
            message, channel_id = await agent.responses.get()

            channel = (
                bot.cache.get_guild_channel(channel_id)
                or await bot.rest.fetch_channel(channel_id)
            )
            
            await channel.send(message)
            
            agent.responses.task_done()

    bot = hikari.GatewayBot(
        intents=hikari.Intents.ALL, 
        token= os.getenv("TOKEN")
    )

    @bot.listen(hikari.GuildMessageCreateEvent)
    async def on_message(event: hikari.GuildMessageCreateEvent):
        """Append messages to the respestive queues allowing the agent to operate on its own."""

        if event.author.id != agent.user_id: 
            if agent.monitoring_channel == event.channel_id:
                await agent.event_queue.put(event)
        
        agent.server.add_message(event.channel_id, event)
                
    @bot.listen(hikari.MemberCreateEvent)
    async def on_member_create(member: hikari.MemberCreateEvent):
        server.update_user(event.author.id, member.display_name)

    @bot.listen(hikari.StoppingEvent)
    async def on_stopping(event: hikari.StoppingEvent) -> None:
        # TODO : Gracefully end tasks.abs
        pass
    
    @bot.listen(hikari.StartedEvent)
    async def on_started(event):
        global agent

        # Create server representation
        server_id = os.getenv("SERVER_ID")
        guild = await bot.rest.fetch_guild(server_id)
        server = DiscordServer(server_id, guild.name)
        
        async for member in bot.rest.fetch_members(server_id):
            server.update_user(member.id, member.display_name)
        
        for channel in await bot.rest.fetch_guild_channels(server_id):
            if isinstance(channel, hikari.TextableChannel):
                server.add_channel(channel.id, channel.name)

        agent = ag.Agent(os.getenv("USER_ID"), agent_conf, server)
  
        print(f"Loaded Server: {server}")
        print(f"Users: {server.users}")
        print(f"Channels: {server.channels}")

        asyncio.create_task(agent.respond_routine())
        asyncio.create_task(agent.plan_routine())
        asyncio.create_task(agent.memory_routine())
        asyncio.create_task(message_handler())
        print(f"Bot is ready")

    bot.run()
