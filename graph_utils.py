import numpy as np
from graph import Graph

def double_edges(adjacency_matrix: np.ndarray, matching: list, shortest_paths: np.ndarray) -> np.ndarray:
    """Double the edges along paths between two matched vertices."""
    new_adjacency_matrix = adjacency_matrix.copy()
    for i, j in matching:
        # Travel along the shortest path between i and j and double the edges
        current_vertex = j
        while current_vertex != i:
            previous_vertex = shortest_paths[i, current_vertex]
            new_adjacency_matrix[current_vertex, previous_vertex] += 1
            new_adjacency_matrix[previous_vertex, current_vertex] += 1
            current_vertex = previous_vertex
    return new_adjacency_matrix