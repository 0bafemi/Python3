import numpy as np
import heapq
from .graph import Cell
from .utils import trace_path

"""
A* search implementation for grid-based path planning.

To visualize which cells are visited in the navigation webapp, save each
visited cell in the list in the graph class as follows:
     graph.visited_cells.append(Cell(cell_i, cell_j))
where cell_i and cell_j are the cell indices of the visited cell you want to
visualize.
"""


def a_star_search(graph, start, goal):
    """A* Search algorithm."""
    # Initialize per-search state in the graph
    graph.init_graph()

    # Heuristic: Euclidean distance from current cell to goal cell
    def h(cell):
        return np.sqrt((cell.i - goal.i) ** 2 + (cell.j - goal.j) ** 2)

    # open_set entries: (f_score, g_score, tie_breaker, Cell)
    counter = 0
    open_set = [(h(start), 0, counter, start)]
    visited = set()

    graph.distance[(start.i, start.j)] = 0
    graph.parent[(start.i, start.j)] = None

    while open_set:
        f, g, _, current = heapq.heappop(open_set)
        key = (current.i, current.j)

        # Skip if we already processed this cell
        if key in visited:
            continue

        visited.add(key)
        graph.visited_cells.append(Cell(current.i, current.j))

        # Goal check
        if current.i == goal.i and current.j == goal.j:
            return trace_path(goal, graph)

        # Explore neighbors
        for neighbor in graph.find_neighbors(current.i, current.j):
            nkey = (neighbor.i, neighbor.j)
            new_g = g + 1  # uniform step cost

            if nkey not in visited and (
                nkey not in graph.distance or new_g < graph.distance[nkey]
            ):
                graph.distance[nkey] = new_g
                graph.parent[nkey] = Cell(current.i, current.j)
                counter += 1
                f_score = new_g + h(neighbor)
                heapq.heappush(open_set, (f_score, new_g, counter, neighbor))

    # No path found
    return []
