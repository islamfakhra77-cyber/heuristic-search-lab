import time
import numpy as np
from typing import Dict, List, Tuple, Callable, Optional
from .grid_world import GridWorld
from .a_star import a_star
from .heuristics import HEURISTICS

class HeuristicEvaluator:
    def __init__(self, world: GridWorld, start: Tuple[int, int], goal: Tuple[int, int]):
        self.world = world
        self.start = start
        self.goal = goal

    def evaluate_single(self, heuristic: Callable, use_optimization: bool = True, num_trials: int = 3):
        expansions_list = []
        time_list = []
        path = None
        cost = None

        for _ in range(num_trials):
            start_time = time.perf_counter()
            path, expansions, cost = a_star(
                self.world, self.start, self.goal,
                heuristic, use_optimization
            )
            elapsed = time.perf_counter() - start_time

            if path is not None:
                expansions_list.append(expansions)
                time_list.append(elapsed)

        return {
            'heuristic': heuristic.__name__,
            'optimization': use_optimization,
            'path_found': path is not None,
            'path_cost': cost,
            'expansions_mean': np.mean(expansions_list) if expansions_list else float('inf'),
            'expansions_std': np.std(expansions_list) if expansions_list else 0,
            'time_mean': np.mean(time_list) * 1000 if time_list else float('inf'),
            'time_std': np.std(time_list) * 1000 if time_list else 0,
            'success_rate': len(expansions_list) / num_trials
        }

    def compare_all(self, heuristic_names: List[str] = None, test_optimization: bool = True):
        if heuristic_names is None:
            heuristic_names = list(HEURISTICS.keys())

        results = []

        for name in heuristic_names:
            heuristic = HEURISTICS[name]

            optimizations = [False, True] if test_optimization else [True]

            for use_opt in optimizations:
                result = self.evaluate_single(heuristic, use_opt)
                results.append(result)

                opt_str = "ON" if use_opt else "OFF"
                print(f"{name:12} | Opt={opt_str} | "
                      f"Expansions: {result['expansions_mean']:6.0f}±{result['expansions_std']:.0f} | "
                      f"Time: {result['time_mean']:6.2f}ms")

        return results

    def test_admissibility(self, heuristic: Callable) -> Tuple[bool, float, float]:
        """
        Test if a heuristic is admissible.

        Args:
            heuristic: Heuristic to test

        Returns:
            Tuple of (is_admissible, optimal_cost, heuristic_cost)
        """
        # Find optimal cost using Dijkstra (zero heuristic)
        _, _, optimal_cost = a_star(
            self.world, self.start, self.goal,
            HEURISTICS['zero'], use_optimization=False
        )

        # Find cost using test heuristic
        _, _, heuristic_cost = a_star(
            self.world, self.start, self.goal,
            heuristic, use_optimization=False
        )

        is_admissible = abs(heuristic_cost - optimal_cost) < 1e-6

        return is_admissible, optimal_cost, heuristic_cost
    