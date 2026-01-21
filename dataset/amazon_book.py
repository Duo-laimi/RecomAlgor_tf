import os
from typing import Dict

import pandas as pd

from .base import DienDataset, Item

"""
-Amazon Book
    - cat_voc.pkl
    - mid_voc.pkl
    - uid_voc.pkl
    - local_train_splitByUser
    - local_test_splitByUser
    - reviews-info
    - item-info
"""

# 从pkl文件中读取类型
def read_from_pkl(path, astype=dict):
    import pickle as pkl
    with open(path, "rb") as f:
        data = pkl.load(f)
    if not isinstance(data, astype):
        raise TypeError("Unmatched Type")
    return data

def read_codebooks(src: str):
    codebooks = {}
    codebook_items = ["cat_voc", "mid_voc", "uid_voc"]
    for item in codebook_items:
        if item.endswith(".pkl"):
            item = item[:-4]
        path = os.path.join(src, item + ".pkl")
        codebooks[item] = read_from_pkl(path, astype=dict)
    return codebooks

def read_meta(path, codebooks):
    cat_voc, mid_voc, uid_voc = list(codebooks.values())
    meta_df = pd.read_csv(path, sep="\t", header=None, names=["mid", "cat"])
    meta = {}
    for row in meta_df.itertuples():
        unique_id = row.mid
        category = row.cat
        mapping_id = mid_voc.get(unique_id, 0)
        category_id = cat_voc.get(category, 0)
        book = Item(unique_id, category, mapping_id, category_id)
        meta[mapping_id] = book
    return meta

def load_amazon_book_df(path: str, codebooks: Dict[str, Dict[str, int]]):
    # src = os.path.dirname(path)
    # codebook_items = ["cat_voc", "mid_voc", "uid_voc"]
    # codebooks = read_codebooks(src, codebook_items)
    cat_voc, mid_voc, uid_voc = list(codebooks.values())

    df = pd.read_csv(path, sep="\t", header=None, names=["label", "uid", "mid", "cat", "mid_str", "cat_str"])
    df["mid_history"] = df["mid_str"].map(lambda x: [mid_voc[item] if item in mid_voc else 0 for item in x.split("\x02")])
    df["cat_history"] = df["cat_str"].map(lambda x: [cat_voc[item] if item in cat_voc else 0 for item in x.split("\x02")])
    df["uid"] = df["uid"].map(lambda item: uid_voc.get(item, 0))
    df["mid"] = df["mid"].map(lambda item: mid_voc.get(item, 0))
    df["cat"] = df["cat"].map(lambda item: cat_voc.get(item, 0))

    return df


def load_amazon_book(source_path: str, codebooks_path: str, meta_path: str):
    codebooks = read_codebooks(codebooks_path)
    meta = read_meta(meta_path, codebooks)
    df = load_amazon_book_df(source_path, codebooks)
    ds = DienDataset(
        source = df,
        meta_mapping = meta,
        codebooks = codebooks,
        padding_length = 16
    )
    return ds
