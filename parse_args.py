import argparse
import logging as log
import os

from utils.common import print_args


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_input", type=str, required=True,
                        help="data_input_dir")
    parser.add_argument("--data_output", type=str, required=True,
                        help="data output path")
    parser.add_argument("--config", type=str, required=True, help="配置文件路径")
    parser.add_argument("--vocab_dir", type=str, help="词典文件夹，可能包含不同序列的item，命名需要一致")

    parser.add_argument("--cache_dir", type=str, help="模型权重文件夹，如果是s3路径，需要先拉取到本地")
    parser.add_argument("--task_mode", type=str, default="train", choices=["train", "inc_train", "eval", "predict"])

    # parser.add_argument("--negative_sample_rate", type=float, default=0.1, help="negative sample rate")

    # 分布式训练参数
    parser.add_argument("--world_size", type=int, default=1, help="number of processes")
    parser.add_argument("--num_workers", type=int, default=4, help="number of workers")
    parser.add_argument("--num_epochs", type=int, default=1, help="number of epochs")
    parser.add_argument("--train_batch_size", type=int, default=32, help="train batch size")
    parser.add_argument("--eval_batch_size", type=int, default=128, help="eval batch size")
    parser.add_argument("--predict_batch_size", type=int, default=128, help="predict batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--warmup_ratio", type=float, default=0.1, help="warmup ratio")
    parser.add_argument("--pin_memory", type=lambda x: x.lower() == 'true', default=False, help="pin memory")

    parser.add_argument("--eval_ratio", type=float, default=0.1, help="eval ratio")
    parser.add_argument("--seed", type=int, default=42, help="random seed")

    args = parser.parse_args()

    print_args(args, log.info)
    return args
