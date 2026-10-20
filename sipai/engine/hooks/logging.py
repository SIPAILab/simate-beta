from typing import Any, Mapping
from sipai.engine.hooks.base import Hook

class HistoryHook(Hook):

    def __init__(self):
        self.history: list[dict[str, Any]] = []

    def after_log(self, trainer: Any, record: Mapping[str, Any]) -> None:
        self.history.append(dict(record))
