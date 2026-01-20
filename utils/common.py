import logging
import logging.config
import os
import time
import zipfile
from functools import wraps
from glob import glob
from types import SimpleNamespace

import yaml

logger = logging.getLogger(__name__)


def print_all_paths(root_dir, handler=print):
    for root, dirs, files in os.walk(root_dir):
        for name in dirs:
            handler(os.path.join(root, name))  # 文件夹路径
        for name in files:
            handler(os.path.join(root, name))  # 文件路径


def setup_logging(config_file):
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    logging.config.dictConfig(config)


def read_yaml(yaml_file):
    with open(yaml_file, "r") as f:
        config = yaml.safe_load(f)
    # 转换为SimpleNamespace
    # config = SimpleNamespace(**config)
    return config


def timer_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()  # 记录开始时间
        result = func(*args, **kwargs)  # 执行函数
        end_time = time.time()  # 记录结束时间
        logger.info(f"函数 {func.__name__} 运行时间: {end_time - start_time:.4f} 秒")
        return result
    return wrapper


def print_args(args, print_func=print):
    print_func("Parsed Arguments:")
    for arg in vars(args):
        print_func(f"{arg:>15}: {getattr(args, arg)}")
