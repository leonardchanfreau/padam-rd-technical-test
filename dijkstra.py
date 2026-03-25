import numpy as np
from graph import Graph

def dijkstra(graph: Graph, start_vertex: int) -> tuple[np.ndarray, np.ndarray]:
    """Implement Dijkstra's algorithm to find the shortest paths from a starting
    vertex to all other vertices in the graph."""
    num_vertices = len(graph.vertices)
    # Initialize result matrices
    distances = np.full(num_vertices, np.inf)
    previous_vertices = np.full(num_vertices, -1, dtype=int)
    # Set the distance to the starting vertex to 0
    distances[start_vertex] = 0
    # Utility matrix to keep track of visited vertices
    visited = np.zeros(num_vertices, dtype=bool)

    for _ in range(num_vertices):
        # Select the unvisited vertex with the smallest distance
        current_vertex = np.argmin(np.where(visited, np.inf, distances))
        visited[current_vertex] = True

        # Update distances to neighboring vertices
        for neighbor in graph.adjacency_matrix[current_vertex].nonzero()[0]:
            # Avoid recomputing distances for visited vertices
            if not visited[neighbor]:
                # Compute the distance passing through the current vertex
                edge_weight = graph.edges[
                    graph.edge_index_map[(current_vertex, neighbor)]][2]
                new_distance = distances[current_vertex] + edge_weight
                # Compare with the previously known distance and update if smaller
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    previous_vertices[neighbor] = current_vertex

    return distances, previous_vertices