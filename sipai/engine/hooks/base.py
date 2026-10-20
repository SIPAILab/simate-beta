from __future__ import annotations
from typing import Any, Mapping

class Hook:

    def before_train(self, trainer: Any) -> None:
        pass

    def after_train(self, trainer: Any) -> None:
        pass

    def before_step(self, trainer: Any, batch: Mapping[str, Any]) -> None:
        pass

    def after_step(self, trainer: Any, batch: Mapping[str, Any], outputs: Mapping[str, Any]) -> None:
        pass

    def after_log(self, trainer: Any, record: Mapping[str, Any]) -> None:
        pass
