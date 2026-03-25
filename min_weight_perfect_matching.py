import numpy as np
from graph import Graph


def _greedy_matching(distances: np.ndarray, vertices_index: list) -> list:
    """Find a greedy perfect matching given the distances between vertices."""
    # Initialize matching list
    matching = []
    # Create a distances matrix that keeps track of edges between unmatched vertices
    remaining_edges = distances.copy()
    # Add infinite weights to the diagonal to avoid matching a vertex with itself
    remaining_edges += np.diag([np.inf] * len(remaining_edges))
    while len(matching) < len(distances) // 2:
        # Find the edge with the smallest weight
        i, j = np.unravel_index(np.argmin(remaining_edges), remaining_edges.shape)
        # Add the edge to the matching
        matching.append((vertices_index[i], vertices_index[j]))
        # Remove the matched vertices from the remaining edges
        remaining_edges[i, :] = np.inf
        remaining_edges[:, i] = np.inf
        remaining_edges[j, :] = np.inf
        remaining_edges[:, j] = np.inf
    return matching


def min_weight_perfect_matching(distances: np.ndarray, vertices_index: list) -> list:
    """Find the minimum weight perfect matching in a complete graph given the
    distances between vertices.

    Parameters
    ----------
    distances : np.ndarray
        A matrix of distances between vertices.
    vertices_index : list
        A list of vertex indices.

    Return :
        matching : list[tuple[int, int]]
            List of matched vertices as tuples (vertex 1, vertex 2).
    """
    # For now we use a greedy solution, which is not optimal
    matching = _greedy_matching(distances, vertices_index)
    # TODO: Implement a more efficient algorithm, such as the Blossom algorithm
    return matching