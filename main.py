import logging as log

import pandas as pd

from parse_args import parse_args
from train_eval.train import train_from_config
from utils.common import setup_logging
from utils.config import Config


def main(args):
    setup_logging(args.log_config)
    logger = log.getLogger(__name__)
    cfg = Config(args.config)
    val_metrics = train_from_config(cfg)
    val_metrics["name"] = cfg["name"]
    df = pd.DataFrame(val_metrics)
    logger.info(df)
    df.to_csv(f"{cfg['name']}.csv")


if __name__ == "__main__":
    argv = parse_args()
    main(argv)