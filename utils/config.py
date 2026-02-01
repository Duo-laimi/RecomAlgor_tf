from typing import Dict, Any

from .common import read_yaml

import logging as log

logger = log.getLogger(__name__)


class Config:
    def __init__(self, config_path):
        # yaml 配置文件
        self.config = read_yaml(config_path)
        self.training_config = self.config["training_args"]
        self.model_config = self.config["model_args"]
        self.data_config = self.config["data_args"]
        self.print_config()

    def print_config(self, print_func=logger.info):
        print_func("Data Args: ")
        for k, v in self.data_config.items():
            print_func(f"{k:>15}: {v}")
        print_func("Training Args: ")
        for k, v in self.training_config.items():
            print_func(f"{k:>15}: {v}")
        print_func("Model Args: ")
        for k, v in self.model_config.items():
            print_func(f"{k:>15}: {v}")

    def update_config(self, new_configs: Dict[str, Any]):
        for key, value in new_configs.items():
            entry = {key: value}
            if key in self.training_config:
                self.training_config.update(entry)
            elif key in self.data_config:
                self.data_config.update(entry)
            elif key in self.model_config:
                self.model_config.update(entry)
            else:
                self.config.update(entry)

if __name__ == "__main__":
    path = "../config/din_config.yaml"
    config = Config(path)
