"""Прім з бінарною купою (heapq) і lazy deletion. На незв'язних графах
повертає кістяковий ліс."""

import heapq
from typing import List, Set, Tuple, Optional

from .base_mst import BaseMST, MSTResult
from models import Edge, Graph


class PrimMST(BaseMST):

    @property
    def algorithm_name(self) -> str:
        return "Prim (Binary Heap)"

    def _find_mst_impl(self, start_node: Optional[int] = None) -> List[Edge]:
        """start_node=None → стартуємо з першої вершини у графі.
        На незв'язному графі повертає кістяковий ліс."""
        if self.graph.node_count() <= 1:
            return []

        if start_node is None:
            start_node = self.graph.get_node_ids()[0]

        mst_edges: List[Edge] = []
        visited: Set[int] = set()
        target_nodes = self.graph.node_count()

        # зовнішній цикл — для незв'язних графів
        while len(visited) < target_nodes:
            if start_node in visited:
                # стартова вже відвідана — беремо першу невідвідану
                for node_id in self.graph.get_node_ids():
                    if node_id not in visited:
                        start_node = node_id
                        break

            visited.add(start_node)

            # Edge має __lt__, тому додатковий counter для heapq не потрібен
            heap: List[Tuple[float, Edge]] = []

            for edge in self.graph.get_neighbors(start_node):
                heapq.heappush(heap, (edge.weight, edge))

            while heap:
                weight, edge = heapq.heappop(heap)

                node1_visited = edge.node1 in visited
                node2_visited = edge.node2 in visited

                if node1_visited and node2_visited:
                    continue  # обидва кінці вже у дереві — пропускаємо

                new_node = edge.node2 if node1_visited else edge.node1
                visited.add(new_node)
                mst_edges.append(edge)
                self._record_step(mst_edges)

                for next_edge in self.graph.get_neighbors(new_node):
                    other_node = next_edge.get_other_node(new_node)
                    if other_node not in visited:
                        heapq.heappush(heap, (next_edge.weight, next_edge))

        return mst_edges


# Приклад використання (для тестування)
if __name__ == "__main__":
    from models import Node, Graph

    # Створення тестового графа
    print("=== Тест алгоритму Пріма ===\n")

    nodes = [
        Node(1, "Підстанція-Центральна", 48.4647, 35.0462),
        Node(2, "Підстанція-Північна", 48.5012, 35.0678),
        Node(3, "Підстанція-Південна", 48.4234, 35.0891),
        Node(4, "Підстанція-Східна", 48.4589, 35.1234),
        Node(5, "Підстанція-Західна", 48.4701, 34.9876),
    ]

    graph = Graph.from_nodes(nodes, cost_per_meter=150.0)

    print(f"Граф: {graph.node_count()} вершин, {graph.edge_count()} ребер")
    print(f"Загальна вартість всіх ребер: {sum(e.weight for e in graph.get_all_edges()):,.2f} грн\n")

    # Пошук МКД
    prim = PrimMST(graph)
    result = prim.find_mst()

    print(result)

    print(f"\nРебра МКД:")
    for i, edge in enumerate(result.edges, 1):
        n1 = graph.get_node(edge.node1).name
        n2 = graph.get_node(edge.node2).name
        print(f"  {i}. {n1} — {n2}: {edge.distance:.2f} м, {edge.weight:,.2f} грн")

    print(f"\nКількість кроків для візуалізації: {len(result.steps)}")