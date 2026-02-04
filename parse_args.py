import argparse
import logging as log
from utils.common import print_args
logger = log.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/din_config.yaml", help="model and training config.")
    parser.add_argument("--param_grid", type=str, default="config/param_grid.yaml", help="parameter grid config.")
    parser.add_argument("--log_config", type=str, default="config/logging_config.yaml")
    args = parser.parse_args()

    print_args(args, logger.info)
    return args
