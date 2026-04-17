import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.grid_world import GridWorld
from src.heuristics import heuristic_manhattan

def test_manhattan_consistency():
    world = GridWorld(5, 5, [])
    goal = (4, 4)
    
    for x in range(5):
        for y in range(5):
            node = (x, y)
            h_node = heuristic_manhattan(node, goal)
            
            for neighbor in world.neighbors(node):
                h_neighbor = heuristic_manhattan(neighbor, goal)
                assert h_node <= h_neighbor + 1, "Manhattan nu satisface inegalitatea triunghiulară"