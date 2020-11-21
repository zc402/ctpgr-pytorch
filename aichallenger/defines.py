from typing import NamedTuple, List


class Box(NamedTuple):
    x1: int
    y1: int
    x2: int
    y2: int


class Joint(NamedTuple):
    x: int
    y: int
    v: int


class Person(NamedTuple):
    box: Box
    joints: List[Joint]


Crowd = List[Person]  # A Crowd refers to all people on image