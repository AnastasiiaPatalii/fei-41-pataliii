"""
Генератор тестових графів для енергомережі.

Підтримує генерацію:
- Випадкових графів з заданою кількістю вершин
- Графів у вигляді сітки (grid)
- Графів з кластерами (імітація реальних мереж)
- Регіональних графів
"""

import random
import math
from typing import Tuple, List, Optional, Dict

from models import Graph, Node, Edge


class GraphGenerator:
    """Генератор тестових графів енергомережі."""

    def __init__(self, seed: Optional[int] = None):
        """
        Ініціалізація генератора.

        Args:
            seed: Seed для генератора випадкових чисел (для відтворюваності)
        """
        if seed is not None:
            random.seed(seed)
        self.seed = seed

    def random_graph(
        self,
        num_nodes: int,
        x_range: Tuple[float, float] = (48.0, 50.0),
        y_range: Tuple[float, float] = (34.0, 36.0),
        cost_per_meter: float = 150.0,
        complete: bool = True
    ) -> Graph:
        """
        Генерує випадковий граф із заданою кількістю вершин.

        Args:
            num_nodes: Кількість підстанцій
            x_range: Діапазон широти (latitude)
            y_range: Діапазон довготи (longitude)
            cost_per_meter: Вартість прокладання за метр
            complete: Якщо True, створює повний граф

        Returns:
            Graph: Згенерований граф
        """
        graph = Graph()

        # Генеруємо вершини з випадковими координатами
        for i in range(num_nodes):
            x = random.uniform(x_range[0], x_range[1])
            y = random.uniform(y_range[0], y_range[1])
            node = Node(
                id=i,
                name=f"Підстанція-{i+1}",
                x=x,
                y=y
            )
            graph.add_node(node)

        # Створюємо повний граф, якщо потрібно
        if complete:
            graph.generate_complete_graph(cost_per_meter)

        return graph

    def grid_graph(
        self,
        rows: int,
        cols: int,
        spacing_km: float = 5.0,
        start_x: float = 49.0,
        start_y: float = 35.0,
        cost_per_meter: float = 150.0
    ) -> Graph:
        """
        Генерує граф у вигляді сітки.

        Args:
            rows: Кількість рядків
            cols: Кількість стовпців
            spacing_km: Відстань між сусідніми вузлами (км)
            start_x: Початкова широта
            start_y: Початкова довгота
            cost_per_meter: Вартість прокладання за метр

        Returns:
            Graph: Граф-сітка
        """
        graph = Graph()

        # Перетворюємо км у градуси (приблизно)
        # 1 градус широти ≈ 111 км
        # 1 градус довготи ≈ 111 * cos(latitude) км
        lat_step = spacing_km / 111.0
        lon_step = spacing_km / (111.0 * math.cos(math.radians(start_x)))

        # Створюємо вершини
        node_id = 0
        for row in range(rows):
            for col in range(cols):
                x = start_x + row * lat_step
                y = start_y + col * lon_step
                node = Node(
                    id=node_id,
                    name=f"Вузол-{row+1}-{col+1}",
                    x=x,
                    y=y
                )
                graph.add_node(node)
                node_id += 1

        # Створюємо ребра тільки між сусідніми вузлами сітки
        for row in range(rows):
            for col in range(cols):
                current_id = row * cols + col
                current_node = graph.nodes[current_id]

                # З'єднуємо з правим сусідом
                if col < cols - 1:
                    right_id = current_id + 1
                    right_node = graph.nodes[right_id]
                    distance = current_node.distance_to(right_node)
                    edge = Edge(
                        node1=current_id,
                        node2=right_id,
                        distance=distance,
                        cost_per_meter=cost_per_meter
                    )
                    graph.add_edge(edge)

                # З'єднуємо з нижнім сусідом
                if row < rows - 1:
                    bottom_id = current_id + cols
                    bottom_node = graph.nodes[bottom_id]
                    distance = current_node.distance_to(bottom_node)
                    edge = Edge(
                        node1=current_id,
                        node2=bottom_id,
                        distance=distance,
                        cost_per_meter=cost_per_meter
                    )
                    graph.add_edge(edge)

        return graph

    def cluster_graph(
        self,
        num_clusters: int,
        nodes_per_cluster: int,
        cluster_radius_km: float = 3.0,
        x_range: Tuple[float, float] = (48.5, 49.5),
        y_range: Tuple[float, float] = (34.5, 35.5),
        cost_per_meter: float = 150.0
    ) -> Graph:
        """
        Генерує граф з кластерами підстанцій.

        Імітує реальну ситуацію, де підстанції групуються
        навколо населених пунктів.

        Args:
            num_clusters: Кількість кластерів (населених пунктів)
            nodes_per_cluster: Кількість підстанцій у кластері
            cluster_radius_km: Радіус кластера (км)
            x_range: Діапазон широти для центрів кластерів
            y_range: Діапазон довготи для центрів кластерів
            cost_per_meter: Вартість прокладання за метр

        Returns:
            Graph: Кластерний граф
        """
        graph = Graph()

        # Генеруємо центри кластерів рівномірно по площі
        cluster_centers = []

        # Ділимо площу на сітку для рівномірного розподілу
        grid_size = math.ceil(math.sqrt(num_clusters))
        x_step = (x_range[1] - x_range[0]) / grid_size
        y_step = (y_range[1] - y_range[0]) / grid_size

        for i in range(num_clusters):
            row = i // grid_size
            col = i % grid_size

            # Додаємо випадкове зміщення в межах комірки
            center_x = x_range[0] + (row + random.uniform(0.2, 0.8)) * x_step
            center_y = y_range[0] + (col + random.uniform(0.2, 0.8)) * y_step

            # Перевіряємо, що не виходимо за межі
            center_x = min(max(center_x, x_range[0]), x_range[1])
            center_y = min(max(center_y, y_range[0]), y_range[1])

            cluster_centers.append((center_x, center_y))

        # Конвертуємо радіус з км у градуси
        radius_deg = cluster_radius_km / 111.0

        # Генеруємо вершини у кожному кластері
        node_id = 0
        for cluster_idx, (cx, cy) in enumerate(cluster_centers):
            for j in range(nodes_per_cluster):
                # Випадкове положення в межах кластера (рівномірний розподіл по колу)
                r = radius_deg * math.sqrt(random.random())  # sqrt для рівномірності по площі
                theta = random.uniform(0, 2 * math.pi)

                x = cx + r * math.cos(theta)
                y = cy + r * math.sin(theta) / math.cos(math.radians(cx))

                node = Node(
                    id=node_id,
                    name=f"ПС-{cluster_idx+1}-{j+1}",
                    x=x,
                    y=y
                )
                graph.add_node(node)
                node_id += 1

        # Створюємо повний граф
        graph.generate_complete_graph(cost_per_meter)

        return graph

    def regional_graph(
        self,
        region: str = "poltava",
        num_nodes: int = 20,
        cost_per_meter: float = 150.0
    ) -> Graph:
        """
        Генерує реалістичний граф для певного регіону України.

        Args:
            region: Назва регіону ('poltava', 'kyiv', 'kharkiv', 'dnipro', 'lviv', 'odesa')
            num_nodes: Кількість підстанцій
            cost_per_meter: Вартість прокладання за метр

        Returns:
            Graph: Регіональний граф
        """
        # Координати регіонів (приблизні центри та розміри)
        regions: Dict[str, Dict] = {
            "poltava": {
                "center_x": 49.5883,
                "center_y": 34.5514,
                "radius_km": 50
            },
            "kyiv": {
                "center_x": 50.4501,
                "center_y": 30.5234,
                "radius_km": 40
            },
            "kharkiv": {
                "center_x": 49.9935,
                "center_y": 36.2304,
                "radius_km": 45
            },
            "dnipro": {
                "center_x": 48.4647,
                "center_y": 35.0462,
                "radius_km": 40
            },
            "lviv": {
                "center_x": 49.8397,
                "center_y": 24.0297,
                "radius_km": 35
            },
            "odesa": {
                "center_x": 46.4825,
                "center_y": 30.7233,
                "radius_km": 45
            }
        }

        if region.lower() not in regions:
            raise ValueError(f"Невідомий регіон: {region}. "
                           f"Доступні: {list(regions.keys())}")

        reg = regions[region.lower()]

        # Конвертуємо радіус у градуси
        radius_deg = reg["radius_km"] / 111.0

        x_range = (reg["center_x"] - radius_deg, reg["center_x"] + radius_deg)
        y_range = (reg["center_y"] - radius_deg, reg["center_y"] + radius_deg)

        return self.random_graph(
            num_nodes=num_nodes,
            x_range=x_range,
            y_range=y_range,
            cost_per_meter=cost_per_meter,
            complete=True
        )

    def sparse_graph(
        self,
        num_nodes: int,
        num_edges: int,
        x_range: Tuple[float, float] = (48.0, 50.0),
        y_range: Tuple[float, float] = (34.0, 36.0),
        cost_per_meter: float = 150.0
    ) -> Graph:
        """Розріджений зв'язний граф з заданою к-стю ребер.

        Backbone-ланцюг 0-1-...-n-1 для зв'язності + випадкові ребра до num_edges.
        """
        if num_edges < num_nodes - 1:
            raise ValueError(
                f"num_edges ({num_edges}) має бути >= num_nodes-1 ({num_nodes-1}) "
                f"для зв'язності"
            )

        graph = Graph()
        for i in range(num_nodes):
            x = random.uniform(x_range[0], x_range[1])
            y = random.uniform(y_range[0], y_range[1])
            graph.add_node(Node(id=i, name=f"Підстанція-{i+1}", x=x, y=y))

        nodes = list(graph.iter_nodes())

        # ланцюг гарантує зв'язність
        for i in range(num_nodes - 1):
            n1, n2 = nodes[i], nodes[i + 1]
            graph.add_edge(Edge(n1.id, n2.id, n1.distance_to(n2), cost_per_meter))

        # Решта ребер — випадкові, без дублікатів
        remaining = num_edges - (num_nodes - 1)
        attempts = 0
        max_attempts = remaining * 20  # щоб не зациклитись на дублікатах
        while remaining > 0 and attempts < max_attempts:
            i, j = random.sample(range(num_nodes), 2)
            if not graph.has_edge(nodes[i].id, nodes[j].id):
                graph.add_edge(Edge(
                    nodes[i].id, nodes[j].id,
                    nodes[i].distance_to(nodes[j]), cost_per_meter
                ))
                remaining -= 1
            attempts += 1

        return graph

    def benchmark_graphs(
        self,
        sizes: List[int] = [10, 50, 100, 200, 500],
        cost_per_meter: float = 150.0
    ) -> List[Tuple[int, Graph]]:
        """
        Генерує набір графів для бенчмаркінгу.

        Args:
            sizes: Список розмірів графів
            cost_per_meter: Вартість прокладання за метр

        Returns:
            List[Tuple[int, Graph]]: Список пар (розмір, граф)
        """
        graphs = []
        for size in sizes:
            graph = self.random_graph(
                num_nodes=size,
                cost_per_meter=cost_per_meter,
                complete=True
            )
            graphs.append((size, graph))
        return graphs