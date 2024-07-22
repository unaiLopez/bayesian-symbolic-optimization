from typing import Optional

class Node:
    def __init__(self, id: str, value: str, left: Optional["Node"] = None, right: Optional["Node"] = None):
        self.id = id
        self.value = value
        self.left = left
        self.right = right