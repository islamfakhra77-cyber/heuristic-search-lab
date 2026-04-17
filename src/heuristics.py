import math

def heuristic_manhattan(node, goal):
    return abs(node[0] - goal[0]) + abs(node[1] - goal[1])

def heuristic_euclidean(node, goal):
    return math.sqrt((node[0] - goal[0])**2 + (node[1] - goal[1])**2)

def heuristic_chebyshev(node, goal):
    return max(abs(node[0] - goal[0]), abs(node[1] - goal[1]))

def heuristic_zero(node, goal):
    return 0.0

HEURISTICS = {
    'zero': heuristic_zero,
    'manhattan': heuristic_manhattan,
    'euclidean': heuristic_euclidean,
    'chebyshev': heuristic_chebyshev
}