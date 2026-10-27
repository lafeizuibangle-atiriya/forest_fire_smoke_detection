"""Pluggable alert channels (console now; email/wechat/sms later)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..core.logging_utils import get_logger


class Notifier(ABC):
    """A single alert channel."""

    @abstractmethod
    def send(self, title: str, message: str, image_path: str | None = None) -> None:
        raise NotImplementedError


class ConsoleNotifier(Notifier):
    """Logs the alert to the console. Useful for local demos."""

    def __init__(self) -> None:
        self._log = get_logger("notifier.console")

    def send(self, title: str, message: str, image_path: str | None = None) -> None:
        self._log.warning("[ALERT] %s | %s | image=%s", title, message, image_path)


def build_notifiers(channels: list[str]) -> list[Notifier]:
    """Factory. Add new channels here without touching call sites."""
    notifiers: list[Notifier] = []
    for channel in channels:
        if channel == "console":
            notifiers.append(ConsoleNotifier())
        else:
            # TODO: implement "email" / "wechat" / "sms" notifiers
            continue
    return notifiers
