from dataset.dien import BaseDataset
from dataset.impl.amazon_book import load_amazon_book

def load_dataset(**data_config) -> BaseDataset:
    return load_amazon_book(**data_config)