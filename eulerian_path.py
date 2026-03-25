import numpy as np
from graph import Graph

def find_eulerian_path(adjacency_matrix: np.ndarray) -> list:
    """Find an eulerian path in a graph represented by its adjacency matrix.
    
    We use the Hierholzer's algorithm.
    
    Return :
    path : list[int]
        List of vertex ids representing the eulerian path."""
    # Step 1: Check if an eulerian path exists
    odd_degree_vertices = \
        np.nonzero(adjacency_matrix.sum(axis=0) % 2 == 1)[0]
    num_odd_degree_vertices = len(odd_degree_vertices)
    if num_odd_degree_vertices > 2:
        raise RuntimeError(
            "The graph must have at most 2 vertices with odd degree to have an \
            eulerian path.")
    # Step 2: Find the starting vertex for the path
    if num_odd_degree_vertices == 2:
        # The graph is semi-eulerian. Start from one of the odd degree vertices
        start_vertex = odd_degree_vertices[0]
    else:
          # The graph is eulerian. Pick any vertex as the starting point
          start_vertex = 0
    # Step 3: initialize the path, the current subtour and the uncovered edges
    path = np.array([start_vertex], dtype=int)
    subtour = [start_vertex]
    uncovered_edges = adjacency_matrix.copy()
    # Step 4: Perform subtours through the graph until we have covered all edges
    while uncovered_edges.sum() > 0:
        current_vertex = subtour[-1]
        # Find a neighbor of the current vertex with an uncovered edge
        uncovered_edges_idx = np.nonzero(uncovered_edges[current_vertex, :])
        if len(uncovered_edges_idx[0]) == 0:
            # We reached the second vertex with an odd degree,
            # the subtour is inserted at the end of the path
            path = np.append(path, subtour[1:])
            # Start a new subtour from any vertex with uncovered edges
            if uncovered_edges.sum() > 0:
                subtour = [path[np.nonzero(uncovered_edges[path].sum(axis=1))[0][0]]]
                continue
        next_vertex = uncovered_edges_idx[0][0]
        # Add the edge to the subtour and mark it as covered
        subtour.append(next_vertex)
        uncovered_edges[current_vertex, next_vertex] -= 1
        uncovered_edges[next_vertex, current_vertex] -= 1
        # If we have returned to the starting vertex, we have completed a subtour
        if next_vertex == subtour[0]:
            # insert the subtour into the main path
            insertion_index = np.where(path == subtour[0])[0][0]
            path = np.insert(path, insertion_index + 1, subtour[1:])
            # Start a new subtour from any vertex with uncovered edges
            if uncovered_edges[path].sum() > 0:
                subtour = [path[np.nonzero(uncovered_edges[path].sum(axis=1))[0][0]]]
            elif uncovered_edges.sum() > 0:
                # TODO: manage the case of two graphs disconnected
                raise RuntimeError(
                    "The remaining of the graph is not connected to the path, no eulerian path exists.")
    return path