import itertools
import os
from typing import List, Dict, Any

from utils.config import Config
import copy


def get_param_grid(**kwargs):
    """
    输入: get_param_grid(learning_rate=[0.01, 0.1], batch_size=[16, 32])
    返回: [
        {'learning_rate': 0.01, 'batch_size': 16},
        {'learning_rate': 0.01, 'batch_size': 32},
        ...
    ]
    """
    # 提取键和对应的值列表
    keys = kwargs.keys()
    values = kwargs.values()

    # 使用 itertools.product 计算笛卡尔积
    # zip(keys, combination) 将参数名与某一组组合重新配对
    grid = [dict(zip(keys, combination)) for combination in itertools.product(*values)]

    return grid

def get_short_name(name):
    items = name.split('_')
    res = []
    for item in items:
        res.append(item[0])
    return "".join(res)

def get_config_generator(init_config: Config, param_settings: List[Dict[str, Any]]):
    for new_config in param_settings:
        current_config = copy.deepcopy(init_config)
        name = init_config['name']
        suffix = [f"{get_short_name(key)}_{value}" for key, value in new_config.items()]
        name = name + "_" + "_".join(suffix)
        new_config["name"] = name
        # 修改save_path和ckpt_path
        new_config["save_path"] = os.path.join(init_config["save_path"], name)
        dirname, filename = os.path.split(init_config["ckpt_path"])
        new_config["ckpt_path"] = os.path.join(dirname, name, filename)
        current_config.update_config(new_config)
        yield current_config