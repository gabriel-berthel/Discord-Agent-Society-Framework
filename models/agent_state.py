from enum import Enum, auto


class AgentState(Enum):
    IDLE = auto()
    PROCESSING = auto()
    READ_ONLY = auto()
    INITIATING_TOPIC = auto()
