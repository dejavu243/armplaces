"""Дополнительные функции"""
from dataclasses import dataclass


def names_swap(_names: dict) -> dict:
    """swap keys in values in dict"""
    return {v: k for k, v in _names.items()}  # key - family, value - position in initial table


def append2lists(lists: list[list], previous_element: str, element: str) -> list[list]:
    """Прикрепить спортсмена к нужной цепочке"""
    lists_upd = []
    for inner_list in lists:
        inner_list_copy = inner_list.copy()
        if inner_list_copy and element not in inner_list_copy and inner_list_copy[-1] == previous_element:
            inner_list_copy.append(element)
        if inner_list_copy not in lists_upd:
            lists_upd.append(inner_list_copy)
    return lists_upd


def drop_duplicates(lists: list[list], cut_less: int = 0) -> list[list]:
    """Удалить дубликаты и отрезать короткие цепочки"""
    lists_upd = []
    for inner_list in lists:
        if inner_list in lists_upd or len(inner_list) <= cut_less:
            continue
        lists_upd.append(inner_list)
    return lists_upd


def get_max_chains(lists: list[list]) -> list[list]:
    """Функция нахождения самых длинных цепочек"""
    max_chain_length = max((len(x) for x in lists), default=0)
    max_chains = []
    for inner_list in lists:
        if len(inner_list) == max_chain_length:
            max_chains.append(inner_list)

    for inner_list in lists:
        if len(inner_list) == max_chain_length - 1 and inner_list not in max_chains:
            max_chains.append(inner_list)

    return max_chains


@dataclass
class TournamentData:
    id: str
    names: dict
    pairs: list
    weights: dict
    with_superfinal: bool = False
