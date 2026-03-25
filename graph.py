from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

Coordinates = tuple[float, float]
Edge = tuple[int, int, int, Coordinates, Coordinates]


class Graph:
    def __init__(
        self,
        vertices: list[Coordinates],
        edges: list[Edge],
    ):
        """Basic constructor of a `Graph` instance.

        Parameters
        ----------
        vertices : list[Coordinates]
            List of vertices coordinates.

        edges : list[Edge]
            List of edges as tuple (id 1, id 2, weight, coordinates 1, coordinates 2).
            
        adjacency_matrix : np.ndarray
            Adjacency matrix of the graph, with 0 if there is no edge between two vertices,
            and the number of edges otherwise.
            
        edge_index_map : dict[tuple[int, int], int]
            A hashmap to easily access the index of an edge given its vertices ids.
        """
        self.vertices = vertices
        self.edges = edges

        # Adjacency matrix of the graph, to simplify further algorithms.
        self.adjacency_matrix = self._compute_adjacency_matrix()
        # edge hashmap to easily access the index of an edge
        self.edge_index_map = {
            (edge[0], edge[1]): i for i, edge in enumerate(self.edges)}
        self.edge_index_map.update(
            {(edge[1], edge[0]): i for i, edge in enumerate(self.edges)}
        )

    def plot(self):
        """
        Plot the graph.
        """
        weights = list(set(edge[2] for edge in self.edges))
        colors = plt.cm.get_cmap("viridis", len(weights))
        _, ax = plt.subplots()
        for i, weight in enumerate(weights):
            lines = [
                [edge[-2][::-1], edge[-1][::-1]]
                for edge in self.edges
                if edge[2] == weight
            ]
            ax.add_collection(
                LineCollection(
                    lines, colors=colors(i), alpha=0.7, label=f"weight {weight}"
                )
            )
        ax.plot()
        ax.legend()
        plt.title(f"#E={len(self.edges)}, #V={len(self.vertices)}")
        plt.show()

    @classmethod
    def display_path(cls, *, path: list[Edge]):
        return cls.display_paths(paths=[path])

    @staticmethod
    def display_paths(*, paths: list[list[Edge]]):
        colors = plt.cm.get_cmap("viridis", len(paths))
        figure = plt.figure()
        ax = figure.add_subplot()
        for path_index, path in enumerate(paths):
            for edge_index, edge in enumerate(path):
                ax.annotate(
                    str(edge_index),
                    xytext=edge[3],
                    xy=edge[4],
                    arrowprops=dict(arrowstyle="->", color=colors(path_index)),
                )
        plt.show()
        
    def _compute_adjacency_matrix(self) -> np.ndarray:
        adjacency_matrix = np.zeros((len(self.vertices), len(self.vertices)), dtype=int)
        for edge in self.edges:
            adjacency_matrix[edge[0], edge[1]] += 1
            adjacency_matrix[edge[1], edge[0]] += 1
        return adjacency_matrix
