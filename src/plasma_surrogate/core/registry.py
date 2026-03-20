"""Simple registry helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Generic, TypeVar


T = TypeVar("T")


@dataclass
class Registry(Generic[T]):
    """A minimal key->factory registry."""

    items: Dict[str, Callable[..., T]] = field(default_factory=dict)

    def register(self, name: str, factory: Callable[..., T]) -> None:
        if name in self.items:
            raise ValueError(f"Registry entry already exists: {name}")
        self.items[name] = factory

    def create(self, name: str, *args, **kwargs) -> T:
        if name not in self.items:
            raise KeyError(f"Unknown registry entry: {name}")
        return self.items[name](*args, **kwargs)

    def has(self, name: str) -> bool:
        return name in self.items
