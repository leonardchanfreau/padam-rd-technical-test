import numpy as np
from graph import Graph
from dijkstra import dijkstra
from min_weight_perfect_matching import min_weight_perfect_matching
from graph_utils import double_edges
from eulerian_path import find_eulerian_path


def eulerian_solution(graph: Graph) -> list:
    """Implement a eulerian solution to find a path in the graph which travels
    all edges.
    This solution is based on the construction of an semi-eulerian graph, by
    doubling edges with the smallest weights."""
    # Step 1: Find vertices with odd degree
    odd_degree_vertices = \
        np.nonzero(graph.adjacency_matrix.sum(axis=0) % 2 == 1)[0]
    # If the graph is already semi-eulerian, we can directly find an eulerian path
    if len(odd_degree_vertices) <= 2:
        adjacency_matrix = graph.adjacency_matrix
    # Otherwise, we need to double some edges to make the graph semi-eulerian
    else:
        # Step 2: Compute the shortest paths between each odd degree vertices
        distances = \
            np.zeros((len(odd_degree_vertices), len(odd_degree_vertices)))
        shortest_paths = \
            -1 * np.ones((len(graph.vertices), len(graph.vertices)), dtype=int)
        for i in range(len(odd_degree_vertices)):
            # TODO: this for loop is very big in cases of large graphs
            # we could optimize it by only computing the shortest paths between
            # odd degree vertices, or by using a more efficient algorithm
            distances_to_i, shortest_paths_from_i = \
                dijkstra(graph, odd_degree_vertices[i])
            distances[i] = distances_to_i[odd_degree_vertices]
            shortest_paths[odd_degree_vertices[i]] = shortest_paths_from_i
        # Step 3: Find the minimum weight perfect matching between those vertices
        matching = min_weight_perfect_matching(distances, odd_degree_vertices)
        # Step 4: Double the edges along the shortest paths of each matching
        # (except the last one, as only a semi-eulerian graph is needed)
        adjacency_matrix = double_edges(
            graph.adjacency_matrix, matching[:-1], shortest_paths)
    # Step 5: Find an eulerian path in the virtual graph
    eulerian_path = find_eulerian_path(adjacency_matrix)
    # Step 6: Convert the path in a correct format
    path = []
    for i, current_vertex in enumerate(eulerian_path[:-1]):
        next_vertex = eulerian_path[i + 1]
        edge = graph.edges[graph.edge_index_map[(current_vertex, next_vertex)]]
        path.append(
            (
                current_vertex,
                next_vertex,
                edge[2],
                graph.vertices[current_vertex],
                graph.vertices[next_vertex],
            )
        )

    return path
