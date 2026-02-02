import itertools
from typing import List, Dict, Any

from utils.config import Config
import copy


def param_grid(**kwargs):
    """
    输入: param_grid(learning_rate=[0.01, 0.1], batch_size=[16, 32])
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

def config_generator(init_config: Config, param_settings: List[Dict[str, Any]]):
    for new_config in param_settings:
        name = f"{init_config['abs_class']}: {new_config}"
        current_config = copy.deepcopy(init_config)
        new_config["name"] = name
        current_config.update_config(new_config)
        yield current_config