# ----- BASE OPTIONS ------

BASE_OPTIONS = {
    "mirostat": 2,
    "mirostat_eta": 0.1,
    "num_ctx": 8192,
}

PENALTY_PROFILE_STRONG = {
    "repeat_penalty": 1.5,
    "presence_penalty": 1.5,
    "frequency_penalty": 0.2,
}

PENALTY_PROFILE_MEDIUM = {
    "repeat_penalty": 1.3,
    "presence_penalty": 1.4,
    "frequency_penalty": 0.2,
}

PENALTY_PROFILE_NONE = {
    "repeat_penalty": 0,
    "presence_penalty": 0,
    "frequency_penalty": 0.7,
}

# ----- OPTIONS DEFINITIONS ------

AGENT_RESPONSE_OPTIONS = {
    **BASE_OPTIONS,
    **PENALTY_PROFILE_STRONG,
    "mirostat_tau": 10,
    "num_predict": 70,
    "penalize_newline": False,
    "stop": ["\n"]
}

AGENT_PLANNING_OPTIONS = {
    **BASE_OPTIONS,
    **PENALTY_PROFILE_STRONG,
    "mirostat_tau": 10,
    "num_predict": 300,
    "newline_penalty": True,
    "stop": ["<|endoftext|>"]
}

CONTEXTUALIZER_NEUTRAL_OPTIONS = {
    **BASE_OPTIONS,
    **PENALTY_PROFILE_NONE,
    "mirostat_tau": 4,
    "num_predict": 300,
    "newline_penalty": True,
    "stop": ["<|endoftext|>"]
}

REFLECTIONS_OPTIONS = {
    **BASE_OPTIONS,
    **PENALTY_PROFILE_STRONG,
    "mirostat_tau": 10,
    "num_predict": 300,
    "newline_penalty": True,
    "stop": ["<|endoftext|>"]
}

QUERIES_OPTIONS = {
    **BASE_OPTIONS,
    **PENALTY_PROFILE_MEDIUM,
    "mirostat_tau": 7,
    "num_predict": 300,
    "stop": ["<|endoftext|>"]
}
