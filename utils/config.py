from typing import Dict, Any

from common import read_yaml


class Config:
    def __init__(self, config_path):
        # yaml 配置文件
        self.config = read_yaml(config_path)
        self.training_config = self.config["training_args"]
        self.model_config = self.config["model_args"]
        self.data_config = self.config["data_args"]
        self.print_config()

    def print_config(self, print_func=print):
        print_func("Data Args: ")
        for k, v in self.data_config.items():
            print_func(f"{k:>15}: {v}")
        print_func("Training Args: ")
        for k, v in self.training_config.items():
            print_func(f"{k:>15}: {v}")
        print_func("Model Args: ")
        for k, v in self.model_config.items():
            print_func(f"{k:>15}: {v}")

    def update_config(self, new_configs: Dict[str, Any], item_name=None):
        if item_name in self.config:
            self.config[item_name].update(new_configs)
        else:
            self.config.update(new_configs)

    def init_metadata(self, num_users, num_items, num_categories):
        meta = {
            "num_users": num_users,
            "num_items": num_items,
            "num_categories": num_categories
        }
        self.update_config(meta, "data_args")


if __name__ == "__main__":
    path = "../config/din_config.yaml"
    config = Config(path)
