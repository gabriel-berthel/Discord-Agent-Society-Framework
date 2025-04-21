from collections import defaultdict, deque
import logging
import os
import pickle

class AgentLogger:
    def __init__(self, persistance_id, log_path, save_logs=False):
        log_level = logging.INFO if save_logs else logging.WARNING
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f"{log_path}/{persistance_id}_log.log")
            ]
        )
        self.logger = logging.getLogger(persistance_id)
        self.logs = defaultdict(list)
        self.log_path = log_path
        self.persistance_id = persistance_id

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
