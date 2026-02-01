from dataclasses import dataclass
from typing import List, Any, Dict, Optional

import numpy as np
import pandas as pd
import tensorflow as tf

from dataset.abstrct import BaseDataset, BaseDatasetLoader


@dataclass
class Item:
    unique_id: str
    category: str
    mapping_id: int
    category_id: int

class DienDataset(BaseDataset):
    def __init__(
            self,
            source: pd.DataFrame,
            meta_mapping: Dict[int, Item],
            codebooks: Dict[str, Dict[str, int]],
            pad_token: Any = 0,
            padding_length: int = 32,
            negative_sample: int = 5,
            negative_prop: Optional[float] = None
    ):
        self.source = source
        self.positive_keys = ["mid_history", "cat_history"]
        self.keys = ["label", "uid", "mid", "cat", "mid_history", "cat_history"]
        self.pad_token = pad_token
        self.padding_length = padding_length

        self.meta_mapping = meta_mapping
        self.sorted_items = [meta_mapping[i] for i in range(len(meta_mapping))]

        self.codebooks = codebooks
        self.codebooks_keys = ["uid_voc", "mid_voc", "cat_voc"]
        self.num_users = len(codebooks["uid_voc"])
        self.num_items = len(codebooks["mid_voc"])
        self.num_categories = len(codebooks["cat_voc"])

        self.negative_prop = negative_prop
        self.negative_sample = negative_sample

    def __len__(self):
        return self.source.shape[0]

    def get_output_signature(self):
        return (
            tf.TensorSpec(shape=(), dtype=tf.float32), # label
            tf.TensorSpec(shape=(), dtype=tf.int32), # uid
            tf.TensorSpec(shape=(), dtype=tf.int32), # mid
            tf.TensorSpec(shape=(), dtype=tf.int32), # cat
            tf.TensorSpec(shape=(self.padding_length,), dtype=tf.int32), # mid_history
            tf.TensorSpec(shape=(self.padding_length,), dtype=tf.int32),  # cat_history
            tf.TensorSpec(shape=(self.padding_length,), dtype=tf.int32),  # positive_mask
            tf.TensorSpec(shape=(self.padding_length, self.negative_sample), dtype=tf.int32), # neg_mid_history
            tf.TensorSpec(shape=(self.padding_length, self.negative_sample), dtype=tf.int32)  # neg_cat_history
        )

    def padding_or_truncate(self, inputs: List[Any]):
        extra = self.padding_length - len(inputs)
        inputs = inputs + [self.pad_token] * extra
        return inputs[-self.padding_length:]

    def negative_sampling(
            self,
            shape: tuple[int, ...],
            positive_items: Optional[np.ndarray] = None,
            sampling_weight: Optional[np.ndarray] = None,
            replace: bool = False
    ):
        """
        根据meta_mapping进行采样，位置编号要与映射编号一致

        :param shape: 返回的采样形状，决定了采样的数量
        :param positive_items: 采样结果中不能包含正样本
        :param sampling_weight: 采样权重
        :param replace: 是否可放回
        :return: 采样结果
        """
        # 将正样本位置的采样权重设置为0
        num_items = len(self.meta_mapping)
        # sampling_weight与self.item_list的顺序一致
        if sampling_weight is None:
            sampling_weight = np.ones(num_items)
        else:
            sampling_weight = sampling_weight.copy()
        if positive_items is not None:
            sampling_weight[positive_items] = 0
        sampling_weight[self.pad_token] = 0
        sampling_weight = sampling_weight / sampling_weight.sum()
        negatives = np.random.choice(num_items, shape, replace=replace, p=sampling_weight)
        return negatives

    def __getitem__(self, idx):
        row = self.source.iloc[idx]
        needed = []
        positives = row["mid_history"]
        num_positives = min(len(positives), self.padding_length)

        for key in self.keys:
            item = row[key]
            if key in self.positive_keys:
                item = self.padding_or_truncate(item)
            item = np.array(item)
            needed.append(item)
        # needed.append(num_positives)
        positive_mask = [1] * num_positives
        positive_mask = self.padding_or_truncate(positive_mask)
        needed.append(np.array(positive_mask, dtype=np.int32))
        sampling_shape = (num_positives, self.negative_sample)
        negatives = self.negative_sampling(sampling_shape, np.array(positives), sampling_weight=self.negative_prop)
        negatives_cat = []
        for idx in negatives.flatten():
            negatives_cat.append(self.meta_mapping[idx].category_id)
        negatives_cat = np.array(negatives_cat, dtype=np.int32).reshape(sampling_shape)
        # 填充
        pad = np.zeros((self.padding_length - num_positives, self.negative_sample))
        negatives = np.concatenate([negatives, pad], axis=0)
        negatives_cat = np.concatenate([negatives_cat, pad], axis=0)

        needed.extend([negatives, negatives_cat])
        # ["label", "uid", "mid", "cat", "mid_history", "cat_history", "positive_mask", "negative_mid_history", "negative_cat_history"]
        return tuple(needed)

class DienDatasetLoader(BaseDatasetLoader):
    def __init__(self, dataset: BaseDataset, shuffle: bool = True):
        self.dataset = dataset
        self.shuffle = shuffle
        self.num_samples = len(dataset)
        self.random_idx = list(range(self.num_samples))
        self.reset()

    def reset(self):
        if self.shuffle:
            self.random_idx = np.random.permutation(self.num_samples).tolist()

    def get_output_signature(self):
        output_signature = self.dataset.get_output_signature()
        return output_signature[1:], output_signature[0]

    def __call__(self):
        for idx in self.random_idx:
            yield self.dataset[idx][1:], self.dataset[idx][0]




