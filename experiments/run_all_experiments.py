import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pandas as pd
from datetime import datetime
from src.grid_world import GridWorld
from src.evaluator import HeuristicEvaluator
from src.visualizer import PathVisualizer

def run_comprehensive_experiments():
    """Run all experiments and save results."""

    # Create results directory
    os.makedirs('../results/data', exist_ok=True)
    os.makedirs('../results/figures', exist_ok=True)

    # Test different maze configurations
    maze_configs = [
        ('empty', 15, (0,0), (14,14)),
        ('sparse', 15, (0,0), (14,14)),
        ('dense', 15, (0,0), (14,14)),
        ('spiral', 15, (0,0), (14,14)),
        ('default', 15, (0,0), (14,14))
    ]

    all_results = {}

    for maze_name, size, start, goal in maze_configs:
        print(f"\n{'='*60}")
        print(f"Testing on {maze_name.upper()} Maze")
        print(f"{'='*60}")

        # Create world
        world = GridWorld.create_maze(maze_name, size)

        # Visualize maze
        PathVisualizer.visualize_world(world, start=start, goal=goal,
                                    title=f"{maze_name.capitalize()} Maze")

        # Run evaluation
        evaluator = HeuristicEvaluator(world, start, goal)
        results = evaluator.compare_all(test_optimization=True)

        # Test admissibility
        print("\n📊 Admissibility Test:")
        for heuristic_name in ['zero', 'manhattan', 'euclidean', 'chebyshev']:
            from src.heuristics import HEURISTICS
            is_adm, opt_cost, h_cost = evaluator.test_admissibility(HEURISTICS[heuristic_name])
            status = "✅" if is_adm else "❌"
            print(f" {status} {heuristic_name:12} | Optimal: {opt_cost} | Found: {h_cost}")

        all_results[maze_name] = results

        # Save results
        df = pd.DataFrame(results)
        df.to_csv(f'../results/data/{maze_name}_results.csv', index=False)

        # Create summary report
        summary = {
            'timestamp': datetime.now().isoformat(),
            'mazes_tested': len(maze_configs),
            'results': all_results
        }

        with open('../results/data/experiment_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

    print("\nAll experiments completed!")
    print("Results saved to ../results/data/")

if __name__ == "__main__":
    run_comprehensive_experiments()