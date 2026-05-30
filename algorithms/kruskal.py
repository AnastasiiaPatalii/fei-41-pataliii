"""Крускал з Union-Find. Сортуємо ребра і беремо найлегші, що не утворюють цикл."""

from typing import List, Optional

from .base_mst import BaseMST, MSTResult
from .union_find import UnionFind
from models import Edge, Graph


class KruskalMST(BaseMST):

    @property
    def algorithm_name(self) -> str:
        return "Kruskal (Union-Find)"

    def _find_mst_impl(self, start_node: Optional[int] = None) -> List[Edge]:
        """start_node ігнорується."""
        if self.graph.node_count() == 0:
            return []

        if self.graph.node_count() == 1:
            return []


        edges = self.graph.get_all_edges()
        edges.sort(key=lambda e: e.weight)

        uf = UnionFind(self.graph.get_node_ids())
        mst_edges: List[Edge] = []
        target_edges = self.graph.node_count() - 1

        for edge in edges:
            # uf.union повертає True якщо кінці у різних компонентах
            if uf.union(edge.node1, edge.node2):
                mst_edges.append(edge)
                self._record_step(mst_edges)
                if len(mst_edges) == target_edges:
                    break

        return mst_edges


# Приклад використання
if __name__ == "__main__":
    from models import Node, Graph

    print("=== Тест алгоритму Крускала ===\n")

    nodes = [
        Node(1, "Підстанція-Центральна", 48.4647, 35.0462),
        Node(2, "Підстанція-Північна", 48.5012, 35.0678),
        Node(3, "Підстанція-Південна", 48.4234, 35.0891),
        Node(4, "Підстанція-Східна", 48.4589, 35.1234),
        Node(5, "Підстанція-Західна", 48.4701, 34.9876),
    ]

    graph = Graph.from_nodes(nodes, cost_per_meter=150.0)

    print(f"Граф: {graph.node_count()} вершин, {graph.edge_count()} ребер\n")

    # Пошук МКД
    kruskal = KruskalMST(graph)
    result = kruskal.find_mst()

    print(result)

    print(f"\nРебра МКД:")
    for i, edge in enumerate(result.edges, 1):
        n1 = graph.get_node(edge.node1).name
        n2 = graph.get_node(edge.node2).name
        print(f"  {i}. {n1} — {n2}: {edge.distance:.2f} м, {edge.weight:,.2f} грн")

    print("\n Алгоритм Крускала працює!")