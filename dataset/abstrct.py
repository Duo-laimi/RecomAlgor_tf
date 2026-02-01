from abc import ABC, abstractmethod
from typing import Any


class BaseDataset(ABC):

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def __getitem__(self, item) -> Any: ...

    @abstractmethod
    def get_output_signature(self): ...

class BaseDatasetLoader(ABC):
    pass