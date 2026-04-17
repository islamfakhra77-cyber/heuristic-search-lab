import numpy as np
from typing import List, Tuple, Set, Optional

class GridWorld:
    def __init__(self, width: int, height: int, obstacles: Optional[List[Tuple[int, int]]] = None):
        self.width = width
        self.height = height
        self.obstacles = set(obstacles) if obstacles else set()

    def is_valid(self, x: int, y: int) -> bool:
        return (0 <= x < self.width and 
                0 <= y < self.height and 
                (x, y) not in self.obstacles)

    def neighbors(self, node: Tuple[int, int]) -> List[Tuple[int, int]]:
        x, y = node
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        return [(x+dx, y+dy) for dx, dy in directions if self.is_valid(x+dx, y+dy)]

    @classmethod
    def create_maze(cls, maze_type: str = "default", size: int = 10):
        if maze_type == "empty":
            return cls(size, size, [])
        elif maze_type == "sparse":
            obstacles = [(i, size//2) for i in range(size//3, 2*size//3)]
            return cls(size, size, obstacles)
        elif maze_type == "dense":
            obstacles = []
            for i in range(size//4, 3*size//4, size//8):
                obstacles.extend([(i, j) for j in range(size)])
            return cls(size, size, obstacles)
        elif maze_type == "spiral":
            obstacles = []
            for i in range(2, size-2):
                obstacles.extend([(i, 2), (i, size-3), (2, i), (size-3, i)])
            return cls(size, size, obstacles)
        else:  # default maze
            obstacles = [
                (3, i) for i in range(size)
            ] + [
                (i, size//2) for i in range(size//2, size)
            ] + [
                (1, 2), (2, 2), (size-3, size-3), (size-2, size-3)
            ]
            return cls(size, size, obstacles)
        