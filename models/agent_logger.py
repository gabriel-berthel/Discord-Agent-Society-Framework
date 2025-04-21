import logging
import os
import pickle
from collections import defaultdict


class AgentLogger:
    def __init__(self, persistance_id: str, log_path: str, log_level=logging.INFO):
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        )
        self.logger: logging.Logger = logging.getLogger(persistance_id)
        self.logs: dict = defaultdict(list)
        self.log_path: str = log_path
        self.persistance_id: str = persistance_id

    def log_event(self, key, input_data, output_data):
        if key not in self.logs.keys():
            self.logs[key] = []

        self.logs[key].append({'input': input_data, 'output': output_data})
        self.logger.debug(f"Agent-Ouput: [key={key}] | Output: {output_data}")

    def save_logs(self):
        os.makedirs(self.log_path, exist_ok=True)
        file_path = os.path.join(self.log_path, f"{self.persistance_id}_log.log")
        with open(file_path, "wb") as f:
            pickle.dump(self.logs, f)

        self.logger.info(f"Saved logs to {file_path}")
