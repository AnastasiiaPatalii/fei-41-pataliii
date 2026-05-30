"""Експорт результатів МКД у JSON / CSV / текстовий звіт / GeoJSON / KML."""

import json
import csv
from datetime import datetime
from typing import Optional, Dict, Any, List, TYPE_CHECKING

from models import Graph, Edge

if TYPE_CHECKING:
    from algorithms.base_mst import MSTResult


class DataExporter:

    @staticmethod
    def to_json(
            result: 'MSTResult',
            graph: Graph,
            path: str,
            algorithm_name: str = "Prim"
    ) -> None:
        """Зберігає результат МКД у JSON файл."""
        data = {
            "metadata": {
                "algorithm": algorithm_name,
                "timestamp": datetime.now().isoformat(),
                "graph_nodes": graph.node_count(),
                "graph_edges": graph.edge_count()
            },
            "summary": {
                "total_cost": result.total_cost,
                "total_distance_m": sum(e.distance for e in result.edges),
                "num_mst_edges": len(result.edges),
                "execution_time_ms": result.execution_time * 1000
            },
            "mst_edges": [
                {
                    "node1": e.node1,
                    "node2": e.node2,
                    "node1_name": graph.get_node(e.node1).name,
                    "node2_name": graph.get_node(e.node2).name,
                    "distance_m": e.distance,
                    "cost_per_meter": e.cost_per_meter,
                    "weight": e.weight
                }
                for e in result.edges
            ],
            "nodes": [
                {
                    "id": n.id,
                    "name": n.name,
                    "x": n.x,
                    "y": n.y
                }
                for n in graph.iter_nodes()
            ]
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def to_csv(
            result: 'MSTResult',
            graph: Graph,
            path: str
    ) -> None:
        """Зберігає ребра МКД у CSV файл."""
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([
                'node1_id', 'node1_name',
                'node2_id', 'node2_name',
                'distance_m', 'cost_per_meter', 'total_cost'
            ])

            for edge in result.edges:
                writer.writerow([
                    edge.node1,
                    graph.get_node(edge.node1).name,
                    edge.node2,
                    graph.get_node(edge.node2).name,
                    f"{edge.distance:.2f}",
                    f"{edge.cost_per_meter:.2f}",
                    f"{edge.weight:.2f}"
                ])

    @staticmethod
    def generate_report(
            result: 'MSTResult',
            graph: Graph,
            algorithm_name: str = "Prim"
    ) -> str:
        """Генерує текстовий звіт про результати МКД."""
        lines = [
            "=" * 60,
            "ЗВІТ: МІНІМАЛЬНЕ КІСТЯКОВЕ ДЕРЕВО ЕНЕРГОМЕРЕЖІ",
            "=" * 60,
            "",
            f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Алгоритм: {algorithm_name}",
            "",
            "-" * 60,
            "ПАРАМЕТРИ ГРАФА",
            "-" * 60,
            f"Кількість підстанцій: {graph.node_count()}",
            f"Кількість можливих з'єднань: {graph.edge_count()}",
            "",
            "-" * 60,
            "РЕЗУЛЬТАТИ ОПТИМІЗАЦІЇ",
            "-" * 60,
            f"Кількість ребер у МКД: {len(result.edges)}",
            f"Загальна довжина мережі: {sum(e.distance for e in result.edges):,.2f} м",
            f"                         ({sum(e.distance for e in result.edges) / 1000:,.2f} км)",
            f"Загальна вартість: {result.total_cost:,.2f} грн",
            f"Час виконання: {result.execution_time * 1000:.3f} мс",
            "",
            "-" * 60,
            "РЕБРА МІНІМАЛЬНОГО КІСТЯКОВОГО ДЕРЕВА",
            "-" * 60,
        ]

        lines.append(f"{'№':<4} {'Підстанція 1':<20} {'Підстанція 2':<20} {'Відстань, м':<12} {'Вартість, грн':<15}")
        lines.append("-" * 75)

        total_distance = 0
        for i, edge in enumerate(result.edges, 1):
            n1_name = graph.get_node(edge.node1).name[:18]
            n2_name = graph.get_node(edge.node2).name[:18]
            lines.append(
                f"{i:<4} {n1_name:<20} {n2_name:<20} {edge.distance:>10,.2f}  {edge.weight:>13,.2f}"
            )
            total_distance += edge.distance

        lines.append("-" * 75)
        lines.append(f"{'РАЗОМ':<45} {total_distance:>10,.2f}  {result.total_cost:>13,.2f}")
        lines.append("")
        lines.append("=" * 60)

        return "\n".join(lines)

    @staticmethod
    def save_benchmark_results(
            results: Dict[str, List],
            path: str
    ) -> None:
        """Зберігає результати бенчмарків."""
        serializable_results = {}

        for algo_name, algo_results in results.items():
            serializable_results[algo_name] = [
                {
                    "algorithm_name": r.algorithm_name,
                    "node_count": r.node_count,
                    "edge_count": r.edge_count,
                    "iterations": r.iterations,
                    "mean_time_ms": r.mean_time * 1000,
                    "std_time_ms": r.std_time * 1000,
                    "median_time_ms": r.median_time * 1000,
                    "min_time_ms": r.min_time * 1000,
                    "max_time_ms": r.max_time * 1000,
                    "mst_cost": r.mst_cost
                }
                for r in algo_results
            ]

        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "benchmarks": serializable_results
            }, f, ensure_ascii=False, indent=2)

    @staticmethod
    def export_for_gis(
            result: 'MSTResult',
            graph: Graph,
            path: str,
            format: str = "geojson"
    ) -> None:
        """Експортує результат у формат для ГІС систем."""
        if format.lower() == "geojson":
            DataExporter._export_geojson(result, graph, path)
        elif format.lower() == "kml":
            DataExporter._export_kml(result, graph, path)
        else:
            raise ValueError(f"Непідтримуваний формат: {format}")

    @staticmethod
    def _export_geojson(result: 'MSTResult', graph: Graph, path: str) -> None:
        """Експорт у GeoJSON."""
        features = []

        for node in graph.iter_nodes():
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [node.y, node.x]
                },
                "properties": {
                    "id": node.id,
                    "name": node.name,
                    "type": "substation"
                }
            })

        for edge in result.edges:
            n1 = graph.get_node(edge.node1)
            n2 = graph.get_node(edge.node2)
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        [n1.y, n1.x],
                        [n2.y, n2.x]
                    ]
                },
                "properties": {
                    "from_id": edge.node1,
                    "to_id": edge.node2,
                    "distance_m": edge.distance,
                    "cost": edge.weight,
                    "type": "power_line"
                }
            })

        geojson = {
            "type": "FeatureCollection",
            "features": features
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(geojson, f, ensure_ascii=False, indent=2)

    @staticmethod
    def _export_kml(result: 'MSTResult', graph: Graph, path: str) -> None:
        """Експорт у KML (Google Earth)."""
        kml_lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<kml xmlns="http://www.opengis.net/kml/2.2">',
            '<Document>',
            '<name>Енергомережа МКД</name>',
            '<description>Мінімальне кістякове дерево енергомережі</description>',
            '',
            '<Style id="substation">',
            '  <IconStyle>',
            '    <Icon><href>http://maps.google.com/mapfiles/kml/shapes/electricity.png</href></Icon>',
            '  </IconStyle>',
            '</Style>',
            '<Style id="powerline">',
            '  <LineStyle>',
            '    <color>ff0000ff</color>',
            '    <width>3</width>',
            '  </LineStyle>',
            '</Style>',
            ''
        ]

        kml_lines.append('<Folder><name>Підстанції</name>')
        for node in graph.iter_nodes():
            kml_lines.extend([
                '<Placemark>',
                f'  <name>{node.name}</name>',
                '  <styleUrl>#substation</styleUrl>',
                '  <Point>',
                f'    <coordinates>{node.y},{node.x},0</coordinates>',
                '  </Point>',
                '</Placemark>'
            ])
        kml_lines.append('</Folder>')

        kml_lines.append('<Folder><name>Лінії електропередач</name>')
        for edge in result.edges:
            n1 = graph.get_node(edge.node1)
            n2 = graph.get_node(edge.node2)
            kml_lines.extend([
                '<Placemark>',
                f'  <name>{n1.name} - {n2.name}</name>',
                f'  <description>Відстань: {edge.distance:.0f} м, Вартість: {edge.weight:.0f} грн</description>',
                '  <styleUrl>#powerline</styleUrl>',
                '  <LineString>',
                '    <coordinates>',
                f'      {n1.y},{n1.x},0',
                f'      {n2.y},{n2.x},0',
                '    </coordinates>',
                '  </LineString>',
                '</Placemark>'
            ])
        kml_lines.append('</Folder>')

        kml_lines.extend([
            '</Document>',
            '</kml>'
        ])

        with open(path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(kml_lines))