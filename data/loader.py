"""Завантажує граф з CSV (nodes + edges) або JSON."""

import csv
import json
import os
from typing import Optional, Dict, Any, List

from models import Graph, Node, Edge


class DataLoader:

    @staticmethod
    def load(path: str, cost_per_meter: float = 150.0) -> Graph:
        """Формат визначаємо за розширенням файлу."""
        if path.endswith('.json'):
            return DataLoader.load_json(path)
        elif path.endswith('.csv'):
            return DataLoader.load_csv(path, cost_per_meter=cost_per_meter)
        else:
            raise ValueError(f"Непідтримуваний формат файлу: {path}")

    @staticmethod
    def load_csv(
            nodes_path: str,
            edges_path: Optional[str] = None,
            cost_per_meter: float = 150.0
    ) -> Graph:
        """
        Завантажує граф з CSV файлів.

        Формат nodes.csv:
            id,name,x,y
            1,Підстанція-1,48.4647,35.0462

        Формат edges.csv (опціонально):
            node1,node2,distance,cost_per_meter
            1,2,4250.5,150
        """
        graph = Graph()

        # Завантажуємо вершини
        with open(nodes_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                node = DataLoader._parse_node(row)
                graph.add_node(node)

        # Завантажуємо ребра або генеруємо повний граф
        if edges_path and os.path.exists(edges_path):
            with open(edges_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    edge = DataLoader._parse_edge(row)
                    graph.add_edge(edge)
        else:
            graph.generate_complete_graph(cost_per_meter)

        return graph

    @staticmethod
    def load_json(path: str) -> Graph:
        """
        Завантажує граф з JSON файлу.

        Формат JSON:
        {
            "config": {
                "default_cost_per_meter": 150.0,
                "distance_formula": "haversine"
            },
            "nodes": [
                {"id": 1, "name": "Підстанція-1", "x": 48.4647, "y": 35.0462}
            ],
            "edges": [
                {"node1": 1, "node2": 2, "distance": 4250.5, "cost_per_meter": 150}
            ]
        }
        """
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        graph = Graph()

        # Отримуємо конфігурацію
        config = data.get('config', {})
        default_cost = config.get('default_cost_per_meter', 150.0)

        # Завантажуємо вершини
        for node_data in data.get('nodes', []):
            node = Node(
                id=int(node_data['id']),
                name=node_data.get('name', f"Node-{node_data['id']}"),
                x=float(node_data['x']),
                y=float(node_data['y'])
            )
            graph.add_node(node)

        # Завантажуємо ребра
        edges_data = data.get('edges', [])
        if edges_data:
            for edge_data in edges_data:
                distance = float(edge_data.get('distance', 0))
                node1_id = int(edge_data['node1'])
                node2_id = int(edge_data['node2'])
                cost = float(edge_data.get('cost_per_meter', default_cost))

                # Обчислюємо відстань ДО створення Edge, щоб уникнути distance=0
                if distance <= 0:
                    n1 = graph.get_node(node1_id)
                    n2 = graph.get_node(node2_id)
                    distance = n1.distance_to(n2)

                edge = Edge(
                    node1=node1_id,
                    node2=node2_id,
                    distance=distance,
                    cost_per_meter=cost
                )
                graph.add_edge(edge)
        else:
            graph.generate_complete_graph(default_cost)

        return graph

    @staticmethod
    def _parse_node(row: Dict[str, str]) -> Node:
        """Парсить рядок CSV у об'єкт Node."""
        return Node(
            id=int(row['id']),
            name=row.get('name', f"Node-{row['id']}"),
            x=float(row['x']),
            y=float(row['y'])
        )

    @staticmethod
    def _parse_edge(row: Dict[str, str]) -> Edge:
        """Парсить рядок CSV у об'єкт Edge."""
        return Edge(
            node1=int(row['node1']),
            node2=int(row['node2']),
            distance=float(row['distance']),
            cost_per_meter=float(row.get('cost_per_meter', 150.0))
        )

    @staticmethod
    def validate_graph(graph: Graph) -> List[str]:
        """
        Перевіряє коректність графа.

        Returns:
            List[str]: Список помилок (порожній, якщо граф коректний)
        """
        errors = []

        if graph.node_count() == 0:
            errors.append("Граф не містить вершин")
            return errors

        # Перевірка ребер
        for edge in graph.iter_edges():
            if not graph.has_node(edge.node1):
                errors.append(f"Ребро посилається на неіснуючу вершину {edge.node1}")
            if not graph.has_node(edge.node2):
                errors.append(f"Ребро посилається на неіснуючу вершину {edge.node2}")
            if edge.weight < 0:
                errors.append(f"Ребро ({edge.node1}, {edge.node2}) має від'ємну вагу")

        # Перевірка зв'язності
        if graph.node_count() > 1 and graph.edge_count() > 0:
            if not graph.is_connected():
                components = graph.get_connected_components()
                errors.append(
                    f"Граф не є зв'язним: {len(components)} компонент(и) зв'язності"
                )

        return errors