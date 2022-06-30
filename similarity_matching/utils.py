from typing import Iterable, TypeVar


T = TypeVar("T")


def flatten(item: Iterable[Iterable[T]]) -> Iterable[T]:
    for i in item:
        for y in i:
            yield y
