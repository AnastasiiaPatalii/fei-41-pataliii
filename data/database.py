"""
SQLite-сховище для мереж, МКД-результатів і бенчмарків.

Таблиці: networks, nodes, edges, mst_results, mst_edges, benchmarks.
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import Optional, List, Tuple, Dict, Any
from contextlib import contextmanager

from models import Node, Edge, Graph
from algorithms import MSTResult


class DatabaseManager:
    """
    Менеджер для роботи з SQLite базою даних.

    Attributes:
        db_path: Шлях до файлу бази даних

    Example:
        >>> db = DatabaseManager("energy.db")
        >>> db.save_network("Тестова", graph)
        >>> networks = db.list_networks()
    """

    def __init__(self, db_path: str = "energy_network.db"):
        """
        Ініціалізація менеджера БД.

        Args:
            db_path: Шлях до файлу SQLite. Створюється автоматично.
        """
        self.db_path = db_path
        self._init_database()

    @contextmanager
    def _connection(self):
        """Контекстний менеджер для з'єднання з БД."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_database(self):
        """Створює таблиці, якщо вони не існують."""
        with self._connection() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS networks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    description TEXT DEFAULT '',
                    cost_per_meter REAL DEFAULT 150.0,
                    distance_method TEXT DEFAULT 'haversine',
                    node_count INTEGER DEFAULT 0,
                    edge_count INTEGER DEFAULT 0,
                    created_at TEXT DEFAULT (datetime('now', 'localtime')),
                    updated_at TEXT DEFAULT (datetime('now', 'localtime'))
                );

                CREATE TABLE IF NOT EXISTS nodes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    network_id INTEGER NOT NULL,
                    node_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    latitude REAL NOT NULL,
                    longitude REAL NOT NULL,
                    metadata TEXT DEFAULT '{}',
                    FOREIGN KEY (network_id) REFERENCES networks(id) ON DELETE CASCADE,
                    UNIQUE(network_id, node_id)
                );

                CREATE TABLE IF NOT EXISTS edges (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    network_id INTEGER NOT NULL,
                    node1_id INTEGER NOT NULL,
                    node2_id INTEGER NOT NULL,
                    distance REAL NOT NULL,
                    cost_per_meter REAL NOT NULL,
                    FOREIGN KEY (network_id) REFERENCES networks(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS mst_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    network_id INTEGER NOT NULL,
                    algorithm TEXT NOT NULL,
                    total_cost REAL,
                    total_distance REAL,
                    edge_count INTEGER,
                    execution_time_ms REAL,
                    peak_memory_kb REAL DEFAULT 0,
                    created_at TEXT DEFAULT (datetime('now', 'localtime')),
                    FOREIGN KEY (network_id) REFERENCES networks(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS mst_edges (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    result_id INTEGER NOT NULL,
                    node1_id INTEGER NOT NULL,
                    node2_id INTEGER NOT NULL,
                    distance REAL NOT NULL,
                    cost_per_meter REAL NOT NULL,
                    weight REAL NOT NULL,
                    FOREIGN KEY (result_id) REFERENCES mst_results(id) ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS benchmarks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    algorithm TEXT NOT NULL,
                    node_count INTEGER,
                    edge_count INTEGER,
                    mean_time_ms REAL,
                    std_time_ms REAL,
                    mst_cost REAL,
                    peak_memory_kb REAL DEFAULT 0,
                    peak_memory_std_kb REAL DEFAULT 0,
                    created_at TEXT DEFAULT (datetime('now', 'localtime'))
                );

                CREATE INDEX IF NOT EXISTS idx_nodes_network ON nodes(network_id);
                CREATE INDEX IF NOT EXISTS idx_edges_network ON edges(network_id);
                CREATE INDEX IF NOT EXISTS idx_mst_results_network ON mst_results(network_id);
                CREATE INDEX IF NOT EXISTS idx_mst_edges_result ON mst_edges(result_id);
            """)

            # Міграція колонок пам'яті у наявних БД. SQLite не має
            # ALTER ... IF NOT EXISTS, тому ловимо OperationalError.
            for table, column, definition in [
                ('mst_results', 'peak_memory_kb', 'REAL DEFAULT 0'),
                ('benchmarks', 'peak_memory_kb', 'REAL DEFAULT 0'),
                ('benchmarks', 'peak_memory_std_kb', 'REAL DEFAULT 0'),
            ]:
                try:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")
                except sqlite3.OperationalError:
                    pass  # колонка вже є

    # ─── Мережі (графи) ───
    # ═══════════════════════════════════════

    def save_network(self, name: str, graph: Graph,
                     cost_per_meter: float = 150.0,
                     distance_method: str = 'haversine',
                     description: str = '') -> int:
        """
        Зберігає граф у базу даних.

        Args:
            name: Назва мережі
            graph: Об'єкт Graph
            cost_per_meter: Вартість за метр (грн/м)
            distance_method: Метод розрахунку відстані
            description: Опис мережі

        Returns:
            ID збереженої мережі
        """
        with self._connection() as conn:
            cursor = conn.execute(
                """INSERT INTO networks (name, description, cost_per_meter,
                   distance_method, node_count, edge_count)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (name, description, cost_per_meter, distance_method,
                 graph.node_count(), graph.edge_count())
            )
            network_id = cursor.lastrowid

            # Вершини
            node_data = [
                (network_id, n.id, n.name, n.x, n.y,
                 json.dumps(getattr(n, 'metadata', None) or {}, ensure_ascii=False))
                for n in graph.iter_nodes()
            ]
            conn.executemany(
                """INSERT INTO nodes (network_id, node_id, name,
                   latitude, longitude, metadata) VALUES (?, ?, ?, ?, ?, ?)""",
                node_data
            )

            # Ребра
            edge_data = [
                (network_id, e.node1, e.node2, e.distance, e.cost_per_meter)
                for e in graph.get_all_edges()
            ]
            conn.executemany(
                """INSERT INTO edges (network_id, node1_id, node2_id,
                   distance, cost_per_meter) VALUES (?, ?, ?, ?, ?)""",
                edge_data
            )

        return network_id

    def load_network(self, network_id: int) -> Optional[Graph]:
        """
        Завантажує граф з бази даних.

        Args:
            network_id: ID мережі

        Returns:
            Об'єкт Graph або None якщо не знайдено
        """
        with self._connection() as conn:
            net = conn.execute(
                "SELECT * FROM networks WHERE id = ?", (network_id,)
            ).fetchone()
            if not net:
                return None

            graph = Graph()

            # Вершини
            rows = conn.execute(
                "SELECT * FROM nodes WHERE network_id = ? ORDER BY node_id",
                (network_id,)
            ).fetchall()
            for r in rows:
                metadata = json.loads(r['metadata']) if r['metadata'] else {}
                node = Node(r['node_id'], r['name'], r['latitude'], r['longitude'])
                graph.add_node(node)

            # Ребра
            rows = conn.execute(
                "SELECT * FROM edges WHERE network_id = ?", (network_id,)
            ).fetchall()
            for r in rows:
                graph.add_edge(Edge(
                    r['node1_id'], r['node2_id'], r['distance'], r['cost_per_meter']
                ))

        return graph

    def list_networks(self) -> List[Dict[str, Any]]:
        """
        Повертає список збережених мереж.

        Returns:
            Список словників з інформацією про мережі
        """
        with self._connection() as conn:
            rows = conn.execute(
                """SELECT id, name, description, cost_per_meter, distance_method,
                   node_count, edge_count, created_at, updated_at
                   FROM networks ORDER BY updated_at DESC"""
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_network(self, network_id: int) -> bool:
        """Видаляє мережу та всі пов'язані дані."""
        with self._connection() as conn:
            cursor = conn.execute(
                "DELETE FROM networks WHERE id = ?", (network_id,)
            )
        return cursor.rowcount > 0

    def update_network(self, network_id: int, graph: Graph,
                       name: str = None, cost_per_meter: float = None) -> bool:
        """Оновлює існуючу мережу."""
        with self._connection() as conn:
            net = conn.execute(
                "SELECT * FROM networks WHERE id = ?", (network_id,)
            ).fetchone()
            if not net:
                return False

            # Видаляємо старі дані
            conn.execute("DELETE FROM nodes WHERE network_id = ?", (network_id,))
            conn.execute("DELETE FROM edges WHERE network_id = ?", (network_id,))

            # Оновлюємо метадані
            conn.execute(
                """UPDATE networks SET
                   name = COALESCE(?, name),
                   cost_per_meter = COALESCE(?, cost_per_meter),
                   node_count = ?, edge_count = ?,
                   updated_at = datetime('now', 'localtime')
                   WHERE id = ?""",
                (name, cost_per_meter, graph.node_count(), graph.edge_count(), network_id)
            )

            # Вставляємо нові дані
            for n in graph.iter_nodes():
                conn.execute(
                    """INSERT INTO nodes (network_id, node_id, name,
                       latitude, longitude, metadata) VALUES (?, ?, ?, ?, ?, ?)""",
                    (network_id, n.id, n.name, n.x, n.y,
                     json.dumps(getattr(n, 'metadata', None) or {}, ensure_ascii=False))
                )
            for e in graph.get_all_edges():
                conn.execute(
                    """INSERT INTO edges (network_id, node1_id, node2_id,
                       distance, cost_per_meter) VALUES (?, ?, ?, ?, ?)""",
                    (network_id, e.node1, e.node2, e.distance, e.cost_per_meter)
                )
        return True

    # ─── Результати МКД ───

    def save_mst_result(self, network_id: int, result: MSTResult,
                        algorithm: str = "Пріма") -> int:
        """
        Зберігає результат побудови МКД.

        Args:
            network_id: ID мережі
            result: Об'єкт MSTResult
            algorithm: Назва алгоритму

        Returns:
            ID збереженого результату
        """
        with self._connection() as conn:
            cursor = conn.execute(
                """INSERT INTO mst_results (network_id, algorithm, total_cost,
                   total_distance, edge_count, execution_time_ms, peak_memory_kb)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (network_id, algorithm, result.total_cost,
                 result.total_distance, result.edge_count,
                 result.execution_time * 1000,
                 result.peak_memory_kb if result.peak_memory_bytes else 0)
            )
            result_id = cursor.lastrowid

            edge_data = [
                (result_id, e.node1, e.node2, e.distance, e.cost_per_meter, e.weight)
                for e in result.edges
            ]
            conn.executemany(
                """INSERT INTO mst_edges (result_id, node1_id, node2_id,
                   distance, cost_per_meter, weight) VALUES (?, ?, ?, ?, ?, ?)""",
                edge_data
            )

        return result_id

    def get_mst_results(self, network_id: int) -> List[Dict[str, Any]]:
        """Повертає всі результати МКД для мережі."""
        with self._connection() as conn:
            rows = conn.execute(
                """SELECT * FROM mst_results WHERE network_id = ?
                   ORDER BY created_at DESC""",
                (network_id,)
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_mst_result(self, result_id: int) -> bool:
        """Видаляє результат МКД."""
        with self._connection() as conn:
            cursor = conn.execute(
                "DELETE FROM mst_results WHERE id = ?", (result_id,)
            )
        return cursor.rowcount > 0

    # ─── Бенчмарки ───

    def save_benchmark(self, algorithm: str, node_count: int, edge_count: int,
                       mean_time_ms: float, std_time_ms: float,
                       mst_cost: float,
                       peak_memory_kb: float = 0.0,
                       peak_memory_std_kb: float = 0.0) -> int:
        """Зберігає результат бенчмарку."""
        with self._connection() as conn:
            cursor = conn.execute(
                """INSERT INTO benchmarks (algorithm, node_count, edge_count,
                   mean_time_ms, std_time_ms, mst_cost,
                   peak_memory_kb, peak_memory_std_kb)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (algorithm, node_count, edge_count, mean_time_ms, std_time_ms,
                 mst_cost, peak_memory_kb, peak_memory_std_kb)
            )
        return cursor.lastrowid

    def get_benchmarks(self, algorithm: str = None) -> List[Dict[str, Any]]:
        """Повертає результати бенчмарків."""
        with self._connection() as conn:
            if algorithm:
                rows = conn.execute(
                    "SELECT * FROM benchmarks WHERE algorithm = ? ORDER BY node_count",
                    (algorithm,)
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM benchmarks ORDER BY created_at DESC, node_count"
                ).fetchall()
        return [dict(r) for r in rows]

    def clear_benchmarks(self) -> int:
        """Очищає всі бенчмарки."""
        with self._connection() as conn:
            cursor = conn.execute("DELETE FROM benchmarks")
        return cursor.rowcount

    # ─── Утиліти ───

    def get_stats(self) -> Dict[str, int]:
        """Повертає статистику бази даних."""
        with self._connection() as conn:
            stats = {}
            for table in ['networks', 'nodes', 'edges', 'mst_results', 'benchmarks']:
                row = conn.execute(f"SELECT COUNT(*) as cnt FROM {table}").fetchone()
                stats[table] = row['cnt']
        return stats

    def export_network_json(self, network_id: int) -> Optional[str]:
        """Експортує мережу у JSON-рядок."""
        with self._connection() as conn:
            net = conn.execute(
                "SELECT * FROM networks WHERE id = ?", (network_id,)
            ).fetchone()
            if not net:
                return None

            nodes = conn.execute(
                "SELECT * FROM nodes WHERE network_id = ?", (network_id,)
            ).fetchall()
            edges = conn.execute(
                "SELECT * FROM edges WHERE network_id = ?", (network_id,)
            ).fetchall()

        data = {
            "config": {
                "name": net['name'],
                "description": net['description'],
                "default_cost_per_meter": net['cost_per_meter'],
                "distance_formula": net['distance_method']
            },
            "nodes": [
                {"id": r['node_id'], "name": r['name'],
                 "x": r['latitude'], "y": r['longitude']}
                for r in nodes
            ],
            "edges": [
                {"node1": r['node1_id'], "node2": r['node2_id'],
                 "distance": r['distance'], "cost_per_meter": r['cost_per_meter']}
                for r in edges
            ]
        }
        return json.dumps(data, ensure_ascii=False, indent=2)

    def import_network_json(self, json_str: str) -> int:
        """Імпортує мережу з JSON-рядка."""
        data = json.loads(json_str)
        cfg = data.get("config", {})

        graph = Graph()
        for nd in data.get("nodes", []):
            graph.add_node(Node(nd['id'], nd['name'], nd['x'], nd['y']))

        cpm = cfg.get("default_cost_per_meter", 150.0)
        edges = data.get("edges", [])
        if edges:
            for ed in edges:
                graph.add_edge(Edge(
                    ed['node1'], ed['node2'], ed['distance'],
                    ed.get('cost_per_meter', cpm)
                ))
        else:
            graph.generate_complete_graph(cpm)

        return self.save_network(
            name=cfg.get("name", "Імпортована мережа"),
            graph=graph,
            cost_per_meter=cpm,
            distance_method=cfg.get("distance_formula", "haversine"),
            description=cfg.get("description", "")
        )


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    db = DatabaseManager("/tmp/test_energy.db")

    # Тест: створення графа
    g = Graph()
    g.add_node(Node(0, "ПС Центр", 48.46, 35.05))
    g.add_node(Node(1, "ПС Північ", 48.51, 35.07))
    g.add_node(Node(2, "ПС Південь", 48.42, 35.03))
    g.generate_complete_graph(150.0)

    # Зберігаємо
    nid = db.save_network("Тестова мережа", g, 150.0, description="Тест SQLite")
    print(f"✅ Збережено мережу ID={nid}")

    # Завантажуємо
    g2 = db.load_network(nid)
    print(f"✅ Завантажено: {g2.node_count()} вершин, {g2.edge_count()} ребер")

    # Список
    nets = db.list_networks()
    print(f"✅ Мереж у БД: {len(nets)}")

    # Статистика
    stats = db.get_stats()
    print(f"✅ Статистика: {stats}")

    # Бенчмарк
    db.save_benchmark("Пріма", 100, 4950, 1.23, 0.15, 500000)
    bm = db.get_benchmarks()
    print(f"✅ Бенчмарків: {len(bm)}")

    # Видаляємо
    db.delete_network(nid)
    print(f"✅ Мережу видалено")

    # Прибираємо
    os.unlink("/tmp/test_energy.db")
    print("\n✅ Всі тести бази даних пройдено!")