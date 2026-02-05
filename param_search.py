from parse_args import parse_args
from train_eval.train import train_from_config
from train_eval.utils import get_param_grid, get_config_generator
from utils.common import setup_logging, read_yaml
from utils.config import Config

import pandas as pd
import logging as log

def main(args):
    setup_logging(args.log_config)
    logger = log.getLogger(__name__)
    init_config = Config(args.config)
    param_grid_config = read_yaml(args.param_grid)
    param_grid = get_param_grid(**param_grid_config)
    cfg_generator = get_config_generator(init_config, param_grid)

    val_metrics_all = []
    for cfg in cfg_generator:
        cfg.print_config()
        val_metrics = train_from_config(cfg)
        val_metrics["name"] = cfg["name"]
        val_metrics_all.append(val_metrics)

    df = pd.DataFrame(val_metrics_all)
    logger.info(df)
    df.to_csv("val_metrics_all.csv")




if __name__ == "__main__":
    argv = parse_args()
    main(argv)



