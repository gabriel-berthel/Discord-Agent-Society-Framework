class Event:

    def __init__(self, channel_id, author_id, display_name, content):
        self.channel_id = channel_id
        self.author_id = author_id
        self.display_name = display_name
        self.content = content

    def __str__(self):
        return f"[{self.display_name}] {self.content}"

    def __repr__(self):
        return (f"Event(channel_id={self.channel_id}, "
                f"author_id={self.author_id}, "
                f"display_name='{self.display_name}', "
                f"content='{self.content}')")