from abc import ABC, abstractmethod
from pathlib import Path


class ChatAgent(ABC):
    def __init__(self, data, savePath, **kwargs):
        if not isinstance(savePath, Path):
            savePath = Path(savePath)
        if savePath.is_dir():
            self._savePath = savePath / savePath.stem
        else:
            raise ValueError('savePath must be a directory.')
        super().__init__()

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def respond(self, prompt):
        return prompt
