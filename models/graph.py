"""Зважений неорієнтований граф енергомережі. Список суміжності."""

from typing import Dict, List, Set, Optional, Iterator, Tuple
from dataclasses import dataclass
from collections import deque
import json

from .node import Node
from .edge import Edge


class GraphError(Exception):
    pass


class NodeNotFoundError(GraphError):
    pass


class EdgeNotFoundError(GraphError):
    pass


class DisconnectedGraphError(GraphError):
    pass


@dataclass
class GraphStats:
    node_count: int
    edge_count: int
    total_distance: float
    total_weight: float
    avg_degree: float
    is_connected: bool
    density: float


class Graph:
    """Неорієнтований граф зі списком суміжності. Ребро (u,v) зберігається
    і для u, і для v."""

    def __init__(self):
        self._nodes: Dict[int, Node] = {}
        self._adjacency: Dict[int, List[Edge]] = {}
        self._edges: Dict[Tuple[int, int], Edge] = {}

    @property
    def nodes(self) -> Dict[int, Node]:
        """Копія словника вершин."""
        return self._nodes.copy()

    @property
    def adjacency(self) -> Dict[int, List[Edge]]:
        """Повертає копію списку суміжності."""
        return {k: list(v) for k, v in self._adjacency.items()}

    def node_count(self) -> int:
        """Повертає кількість вершин."""
        return len(self._nodes)

    def edge_count(self) -> int:
        """Повертає кількість ребер."""
        return len(self._edges)

    def is_empty(self) -> bool:
        """Перевіряє, чи граф порожній."""
        return len(self._nodes) == 0

    # --- Вершини ---

    def add_node(self, node: Node) -> None:
        """Додає вершину до графа."""
        if node.id in self._nodes:
            raise ValueError(f"Вершина з ID {node.id} вже існує в графі")

        self._nodes[node.id] = node
        self._adjacency[node.id] = []

    def remove_node(self, node_id: int) -> Node:
        """Видаляє вершину та всі інцидентні їй ребра."""
        if node_id not in self._nodes:
            raise NodeNotFoundError(f"Вершина з ID {node_id} не знайдена")

        edges_to_remove = list(self._adjacency[node_id])
        for edge in edges_to_remove:
            self._remove_edge_internal(edge)

        node = self._nodes.pop(node_id)
        del self._adjacency[node_id]

        return node

    def get_node(self, node_id: int) -> Node:
        """Повертає вершину за ID."""
        if node_id not in self._nodes:
            raise NodeNotFoundError(f"Вершина з ID {node_id} не знайдена")
        return self._nodes[node_id]

    def has_node(self, node_id: int) -> bool:
        """Перевіряє наявність вершини в графі."""
        return node_id in self._nodes

    def get_node_ids(self) -> List[int]:
        """Повертає список ID всіх вершин."""
        return list(self._nodes.keys())

    def iter_nodes(self) -> Iterator[Node]:
        """Ітератор по всіх вершинах."""
        return iter(self._nodes.values())

    # --- Ребра ---

    def add_edge(self, edge: Edge) -> None:
        """Додає ребро до графа."""
        if edge.node1 not in self._nodes:
            raise NodeNotFoundError(f"Вершина {edge.node1} не знайдена в графі")
        if edge.node2 not in self._nodes:
            raise NodeNotFoundError(f"Вершина {edge.node2} не знайдена в графі")

        edge_key = (edge.node1, edge.node2)
        if edge_key in self._edges:
            raise ValueError(f"Ребро {edge_key} вже існує в графі")

        self._adjacency[edge.node1].append(edge)
        self._adjacency[edge.node2].append(edge)
        self._edges[edge_key] = edge

    def remove_edge(self, node1: int, node2: int) -> Edge:
        """Видаляє ребро між двома вершинами."""
        if node1 > node2:
            node1, node2 = node2, node1

        edge_key = (node1, node2)
        if edge_key not in self._edges:
            raise EdgeNotFoundError(f"Ребро ({node1}, {node2}) не знайдене")

        edge = self._edges[edge_key]
        self._remove_edge_internal(edge)
        return edge

    def _remove_edge_internal(self, edge: Edge) -> None:
        """Внутрішній метод видалення ребра."""
        edge_key = (edge.node1, edge.node2)

        if edge_key in self._edges:
            del self._edges[edge_key]

        if edge.node1 in self._adjacency:
            self._adjacency[edge.node1] = [e for e in self._adjacency[edge.node1]
                                           if e != edge]
        if edge.node2 in self._adjacency:
            self._adjacency[edge.node2] = [e for e in self._adjacency[edge.node2]
                                           if e != edge]

    def get_edge(self, node1: int, node2: int) -> Edge:
        """Повертає ребро між двома вершинами."""
        if node1 > node2:
            node1, node2 = node2, node1

        edge_key = (node1, node2)
        if edge_key not in self._edges:
            raise EdgeNotFoundError(f"Ребро ({node1}, {node2}) не знайдене")

        return self._edges[edge_key]

    def has_edge(self, node1: int, node2: int) -> bool:
        """Перевіряє наявність ребра між двома вершинами."""
        if node1 > node2:
            node1, node2 = node2, node1
        return (node1, node2) in self._edges

    def get_neighbors(self, node_id: int) -> List[Edge]:
        """Повертає список ребер, інцидентних вершині."""
        if node_id not in self._nodes:
            raise NodeNotFoundError(f"Вершина {node_id} не знайдена")
        return list(self._adjacency[node_id])

    def get_neighbor_ids(self, node_id: int) -> List[int]:
        """Повертає список ID сусідніх вершин."""
        edges = self.get_neighbors(node_id)
        return [edge.get_other_node(node_id) for edge in edges]

    def get_degree(self, node_id: int) -> int:
        """Повертає степінь вершини."""
        return len(self.get_neighbors(node_id))

    def iter_edges(self) -> Iterator[Edge]:
        """Ітератор по всіх ребрах."""
        return iter(self._edges.values())

    def get_all_edges(self) -> List[Edge]:
        """Повертає список всіх ребер."""
        return list(self._edges.values())

    # --- Генерація ---

    def generate_complete_graph(self, cost_per_meter: float = 100.0,
                                distance_method: str = 'haversine') -> None:
        """Робить граф повним (всі пари вершин з'єднано)."""
        self._edges.clear()
        for node_id in self._adjacency:
            self._adjacency[node_id] = []

        node_ids = list(self._nodes.keys())
        for i, id1 in enumerate(node_ids):
            for id2 in node_ids[i + 1:]:
                node1 = self._nodes[id1]
                node2 = self._nodes[id2]

                distance = node1.distance_to(node2, method=distance_method)
                edge = Edge(id1, id2, distance=distance, cost_per_meter=cost_per_meter)

                self._adjacency[id1].append(edge)
                self._adjacency[id2].append(edge)
                self._edges[(edge.node1, edge.node2)] = edge

    # --- Аналіз ---

    def is_connected(self) -> bool:
        """Перевіряє, чи граф є зв'язним (BFS)."""
        if self.node_count() <= 1:
            return True

        start_node = next(iter(self._nodes))
        visited = self._bfs(start_node)

        return len(visited) == self.node_count()

    def _bfs(self, start_node: int) -> Set[int]:
        """Обхід в ширину (BFS) від заданої вершини."""
        visited: Set[int] = {start_node}
        queue = deque([start_node])

        while queue:
            current = queue.popleft()
            for edge in self._adjacency[current]:
                neighbor = edge.get_other_node(current)
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)

        return visited

    def get_connected_components(self) -> List[Set[int]]:
        """Знаходить всі компоненти зв'язності."""
        visited: Set[int] = set()
        components: List[Set[int]] = []

        for node_id in self._nodes:
            if node_id not in visited:
                component = self._bfs(node_id)
                visited.update(component)
                components.append(component)

        return components

    def get_stats(self) -> GraphStats:
        """Обчислює статистику графа."""
        node_count = self.node_count()
        edge_count = self.edge_count()

        total_distance = sum(e.distance for e in self._edges.values())
        total_weight = sum(e.weight for e in self._edges.values())

        avg_degree = (2 * edge_count / node_count) if node_count > 0 else 0

        max_edges = node_count * (node_count - 1) / 2 if node_count > 1 else 1
        density = edge_count / max_edges if max_edges > 0 else 0

        return GraphStats(
            node_count=node_count,
            edge_count=edge_count,
            total_distance=total_distance,
            total_weight=total_weight,
            avg_degree=avg_degree,
            is_connected=self.is_connected(),
            density=density
        )

    # --- Серіалізація ---

    def to_dict(self) -> dict:
        """Серіалізує граф у словник."""
        return {
            'nodes': [node.to_dict() for node in self._nodes.values()],
            'edges': [edge.to_dict() for edge in self._edges.values()]
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Graph':
        """Створює граф зі словника."""
        graph = cls()

        for node_data in data['nodes']:
            graph.add_node(Node.from_dict(node_data))

        for edge_data in data['edges']:
            edge_data_clean = {k: v for k, v in edge_data.items() if k != 'weight'}
            graph.add_edge(Edge.from_dict(edge_data_clean))

        return graph

    def to_json(self, path: str, indent: int = 2) -> None:
        """Зберігає граф у JSON файл."""
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=indent)

    @classmethod
    def from_json(cls, path: str) -> 'Graph':
        """Завантажує граф з JSON файлу."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)

    # --- Фабричні методи ---

    @classmethod
    def from_nodes(cls, nodes: List[Node],
                   cost_per_meter: float = 100.0,
                   distance_method: str = 'haversine') -> 'Graph':
        """Створює повний граф зі списку вершин."""
        graph = cls()
        for node in nodes:
            graph.add_node(node)
        graph.generate_complete_graph(cost_per_meter, distance_method)
        return graph

    @classmethod
    def from_nodes_and_edges(cls, nodes: List[Node], edges: List[Edge]) -> 'Graph':
        """Створює граф зі списків вершин і ребер."""
        graph = cls()
        for node in nodes:
            graph.add_node(node)
        for edge in edges:
            graph.add_edge(edge)
        return graph

    # --- dunder-методи ---

    def __len__(self) -> int:
        """Повертає кількість вершин."""
        return self.node_count()

    def __contains__(self, item) -> bool:
        """Перевіряє наявність вершини або ребра."""
        if isinstance(item, int):
            return self.has_node(item)
        elif isinstance(item, Node):
            return self.has_node(item.id)
        elif isinstance(item, Edge):
            return self.has_edge(item.node1, item.node2)
        elif isinstance(item, tuple) and len(item) == 2:
            return self.has_edge(item[0], item[1])
        return False

    def __repr__(self) -> str:
        """Детальне представлення для відладки."""
        return f"Graph(nodes={self.node_count()}, edges={self.edge_count()})"

    def __str__(self) -> str:
        """Читабельне представлення."""
        stats = self.get_stats()
        connected = "зв'язний" if stats.is_connected else "незв'язний"
        return (f"Граф енергомережі: {stats.node_count} підстанцій, "
                f"{stats.edge_count} ліній ({connected})\n"
                f"Загальна довжина: {stats.total_distance:,.2f} м\n"
                f"Загальна вартість: {stats.total_weight:,.2f} грн")


# Приклад використання
if __name__ == "__main__":
    # Створення підстанцій у Дніпрі
    nodes = [
        Node(1, "Підстанція-Центральна", 48.4647, 35.0462),
        Node(2, "Підстанція-Північна", 48.5012, 35.0678),
        Node(3, "Підстанція-Південна", 48.4234, 35.0891),
        Node(4, "Підстанція-Східна", 48.4589, 35.1234),
        Node(5, "Підстанція-Західна", 48.4701, 34.9876),
    ]

    # Створення повного графа
    graph = Graph.from_nodes(nodes, cost_per_meter=150.0)

    print("=== Граф енергомережі ===")
    print(graph)

    print("\n=== Статистика ===")
    stats = graph.get_stats()
    print(f"Щільність графа: {stats.density:.2%}")
    print(f"Середній степінь: {stats.avg_degree:.2f}")

    print("\n=== Сусіди вершини 1 ===")
    for edge in graph.get_neighbors(1):
        other = edge.get_other_node(1)
        print(f"  → {graph.get_node(other).name}: {edge.distance:.2f} м")

    print("\n=== Всі ребра (відсортовані за вагою) ===")
    edges_sorted = sorted(graph.get_all_edges())
    for edge in edges_sorted[:5]:  # Топ-5 найдешевших
        print(f"  {edge}")

    # Збереження в JSON
    graph.to_json('/tmp/test_graph.json')
    print("\n=== Збережено в /tmp/test_graph.json ===")

    # Завантаження з JSON
    loaded = Graph.from_json('/tmp/test_graph.json')
    print(f"Завантажено: {loaded}")