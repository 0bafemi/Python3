import argparse
from src.graph import GridGraph, Cell
from src.graph_search import a_star_search
from src.utils import generate_plan_file


def parse_args():
    parser = argparse.ArgumentParser(description="HelloRob Path Planning Client.")
    parser.add_argument("-m", "--map", type=str, required=True, help="Path to the map file.")
    parser.add_argument("--start", type=int, nargs=2, required=True, help="Start cell.")
    parser.add_argument("--goal", type=int, nargs=2, required=True, help="Goal cell.")

    # We no longer expose an algorithm choice; we always use A*.
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()

    # Construct the graph.
    graph = GridGraph(args.map)
    # Construct the start and goal cells.
    start, goal = Cell(*args.start), Cell(*args.goal)

    # Always run A* search.
    path = a_star_search(graph, start, goal)

    # Output the planning file for visualization.
    generate_plan_file(graph, start, goal, path, algo="astar")
