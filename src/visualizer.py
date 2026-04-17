import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, Circle
from typing import List, Tuple, Optional, Dict
from .grid_world import GridWorld

class PathVisualizer:
    @staticmethod
    def visualize_world(
        world: GridWorld,
        path: Optional[List[Tuple[int, int]]] = None,
        start: Optional[Tuple[int, int]] = None,
        goal: Optional[Tuple[int, int]] = None,
        title: str = "Grid World",
        figsize: Tuple[int, int] = (10, 10)
    ):
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Draw grid cells
        for x in range(world.width):
            for y in range(world.height):
                if (x, y) in world.obstacles:
                    rect = Rectangle((x, y), 1, 1, facecolor='black', edgecolor='gray')
                    ax.add_patch(rect)
                else:
                    rect = Rectangle((x, y), 1, 1, facecolor='white', edgecolor='lightgray')
                    ax.add_patch(rect)

        # Draw path if provided
        if path:
            path_x = [p[0] + 0.5 for p in path]
            path_y = [p[1] + 0.5 for p in path]
            ax.plot(path_x, path_y, 'b-', linewidth=3, alpha=0.7, label='Path')
            ax.scatter(path_x, path_y, c='blue', s=50, alpha=0.7)

        # Draw start
        if start:
            circle = Circle((start[0] + 0.5, start[1] + 0.5), 0.3,
                          facecolor='green', edgecolor='darkgreen', linewidth=2)
            ax.add_patch(circle)
            ax.text(start[0] + 0.5, start[1] + 0.5, 'S',
                   ha='center', va='center', fontweight='bold', color='white')

        # Draw goal
        if goal:
            circle = Circle((goal[0] + 0.5, goal[1] + 0.5), 0.3,
                          facecolor='red', edgecolor='darkred', linewidth=2)
            ax.add_patch(circle)
            ax.text(goal[0] + 0.5, goal[1] + 0.5, 'G',
                   ha='center', va='center', fontweight='bold', color='white')

        ax.set_xlim(0, world.width)
        ax.set_ylim(0, world.height)
        ax.set_xticks(range(world.width + 1))
        ax.set_yticks(range(world.height + 1))
        ax.grid(True, linestyle='--', alpha=0.5)
        ax.set_aspect('equal')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend(loc='upper right')

        plt.tight_layout()
        plt.show()

    @staticmethod
    def visualize_heuristic(
        world: GridWorld,
        heuristic,
        goal: Tuple[int, int],
        title: str = "Heuristic Heatmap",
        figsize: Tuple[int, int] = (10, 8)
    ):
        heatmap = np.zeros((world.height, world.width))

        for x in range(world.width):
            for y in range(world.height):
                if world.is_valid(x, y):
                    heatmap[y, x] = heuristic((x, y), goal)
                else:
                    heatmap[y, x] = np.nan

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        im = ax.imshow(heatmap, origin='lower', cmap='viridis', alpha=0.8)

        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Heuristic Value', fontsize=12)

        if goal:
            ax.plot(goal[0], goal[1], 'r*', markersize=15, label='Goal')

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        ax.legend()

        plt.tight_layout()
        plt.show()