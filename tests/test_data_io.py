"""
test_data_io.py — Інтеграційні тести введення/виведення даних

Рівень тестування: Integration (інтеграційне)
Мета: Перевірити коректність збереження та завантаження даних через
      усі канали системи:
      - SQLite (DatabaseManager): збереження мереж, результатів МКД, бенчмарків
      - JSON (DataExporter / DataLoader): експорт та імпорт
      - CSV (DataExporter): експорт результатів

Методологія:
    Round-trip тестування: дані зберігаються → завантажуються → порівнюються
    з оригіналом. Якщо збіглися — канал I/O не спотворює дані.

Також перевіряємо:
    - GraphGenerator: валідність згенерованих графів
    - DataLoader.validate_graph: виявлення помилок
    - Пакетне збереження з історією (сценарій наукового керівника)
"""

import pytest
import os
import json
import csv
from models import Node, Edge, Graph
from data.database import DatabaseManager
from data.loader import DataLoader
from data.exporter import DataExporter
from data.generator import GraphGenerator
from algorithms import PrimMST, KruskalMST


# ФІКСТУРИ

@pytest.fixture
def sample_graph():
    """
    Еталонний граф для тестів серіалізації.
    3 підстанції, 2 ребра, відомі ваги.
    """
    g = Graph()
    g.add_node(Node(1, "ПС-Центр", 48.5, 35.1, metadata={"type": "main"}))
    g.add_node(Node(2, "ПС-Північ", 48.6, 35.2))
    g.add_node(Node(3, "ПС-Південь", 48.7, 35.3))
    g.add_edge(Edge(1, 2, distance=15000.5, cost_per_meter=120.0))
    g.add_edge(Edge(2, 3, distance=12000.0, cost_per_meter=150.0))
    return g


@pytest.fixture
def temp_db(tmp_path):
    """Тимчасова SQLite БД (автоматично видаляється після тесту)."""
    db_file = str(tmp_path / "test_energy.db")
    yield DatabaseManager(db_file)


@pytest.fixture
def temp_json(tmp_path):
    """Тимчасовий шлях для JSON файлу."""
    return str(tmp_path / "test_export.json")


@pytest.fixture
def temp_csv(tmp_path):
    """Тимчасовий шлях для CSV файлу."""
    return str(tmp_path / "test_export.csv")


# ГЕНЕРАТОР ГРАФІВ

class TestGraphGenerator:
    """
    Тести генератора тестових графів.
    Генератор використовується для бенчмарків і крос-валідації,
    тому його коректність критична.
    """

    def test_random_graph_node_count(self):
        """Кількість вершин відповідає параметру num_nodes."""
        gen = GraphGenerator(seed=42)
        g = gen.random_graph(num_nodes=25, complete=True)
        assert g.node_count() == 25

    def test_random_complete_graph_edge_count(self):
        """Повний граф K_n має n(n-1)/2 ребер."""
        gen = GraphGenerator(seed=42)
        n = 10
        g = gen.random_graph(num_nodes=n, complete=True)
        assert g.edge_count() == n * (n - 1) // 2

    def test_random_graph_is_connected(self):
        """Повний випадковий граф завжди зв'язний."""
        gen = GraphGenerator(seed=42)
        g = gen.random_graph(num_nodes=15, complete=True)
        assert g.is_connected()

    def test_random_graph_no_self_loops(self):
        """У згенерованому графі немає петель (self-loops)."""
        gen = GraphGenerator(seed=42)
        g = gen.random_graph(num_nodes=20, complete=True)
        for edge in g.iter_edges():
            assert edge.node1 != edge.node2

    def test_random_graph_positive_distances(self):
        """Усі відстані в згенерованому графі додатні."""
        gen = GraphGenerator(seed=42)
        g = gen.random_graph(num_nodes=10, complete=True)
        for edge in g.iter_edges():
            assert edge.distance > 0

    def test_grid_graph_structure(self):
        """Сітка 3×3 має 9 вершин і 12 ребер (горизонтальні + вертикальні)."""
        gen = GraphGenerator(seed=1)
        g = gen.grid_graph(rows=3, cols=3)
        assert g.node_count() == 9
        # 3 ряди × 2 горизонтальні + 2 ряди × 3 вертикальні = 6 + 6 = 12
        assert g.edge_count() == 12

    def test_grid_graph_is_connected(self):
        """Сітковий граф завжди зв'язний."""
        gen = GraphGenerator(seed=1)
        g = gen.grid_graph(rows=4, cols=4)
        assert g.is_connected()

    def test_cluster_graph_node_count(self):
        """Кластерний граф: 3 кластери × 5 вузлів = 15 вузлів."""
        gen = GraphGenerator(seed=42)
        g = gen.cluster_graph(num_clusters=3, nodes_per_cluster=5)
        assert g.node_count() == 15

    def test_cluster_graph_is_connected(self):
        """Кластерний граф зв'язний (кластери з'єднані між собою)."""
        gen = GraphGenerator(seed=42)
        g = gen.cluster_graph(num_clusters=3, nodes_per_cluster=5)
        assert g.is_connected()

    def test_seed_reproducibility(self):
        """Однаковий seed дає ідентичний граф (відтворюваність)."""
        g1 = GraphGenerator(seed=42).random_graph(num_nodes=10, complete=True)
        g2 = GraphGenerator(seed=42).random_graph(num_nodes=10, complete=True)
        assert g1.node_count() == g2.node_count()
        assert g1.edge_count() == g2.edge_count()

        # Порівнюємо координати першої вершини
        n1 = g1.get_node(0)
        n2 = g2.get_node(0)
        assert n1.x == pytest.approx(n2.x)
        assert n1.y == pytest.approx(n2.y)

    def test_different_seeds_different_graphs(self):
        """Різні seed-и дають різні графи."""
        g1 = GraphGenerator(seed=1).random_graph(num_nodes=10, complete=True)
        g2 = GraphGenerator(seed=2).random_graph(num_nodes=10, complete=True)
        n1 = g1.get_node(0)
        n2 = g2.get_node(0)
        # Вкрай малоймовірно, що координати збіглися
        assert n1.x != pytest.approx(n2.x) or n1.y != pytest.approx(n2.y)

    def test_coordinates_in_ukraine_bounds(self):
        """Координати генеруються в межах Полтавської області."""
        gen = GraphGenerator(seed=42)
        g = gen.random_graph(num_nodes=50, complete=False)
        for node in g.iter_nodes():
            assert 44.0 <= node.x <= 53.0, f"Широта {node.x} за межами"
            assert 22.0 <= node.y <= 41.0, f"Довгота {node.y} за межами"

# БАЗА ДАНИХ SQLite — ЗБЕРЕЖЕННЯ ТА ЗАВАНТАЖЕННЯ МЕРЕЖІ
class TestDatabaseNetwork:
    """
    Тести цілісності БД: Граф → SQLite → Граф.
    Доводимо, що база даних не спотворює структуру та числові дані.
    """

    def test_save_returns_positive_id(self, sample_graph, temp_db):
        """Збереження повертає додатній ID мережі."""
        net_id = temp_db.save_network("Тестова", sample_graph, cost_per_meter=150.0)
        assert net_id > 0

    def test_round_trip_node_count(self, sample_graph, temp_db):
        """Кількість вершин зберігається."""
        net_id = temp_db.save_network("Тест", sample_graph)
        loaded = temp_db.load_network(net_id)
        assert loaded.node_count() == sample_graph.node_count()

    def test_round_trip_edge_count(self, sample_graph, temp_db):
        """Кількість ребер зберігається."""
        net_id = temp_db.save_network("Тест", sample_graph)
        loaded = temp_db.load_network(net_id)
        assert loaded.edge_count() == sample_graph.edge_count()

    def test_round_trip_coordinates(self, sample_graph, temp_db):
        """Координати вершин зберігаються з точністю float64."""
        net_id = temp_db.save_network("Тест", sample_graph)
        loaded = temp_db.load_network(net_id)

        orig = sample_graph.get_node(1)
        rest = loaded.get_node(1)
        assert rest.name == orig.name
        assert rest.x == pytest.approx(orig.x)
        assert rest.y == pytest.approx(orig.y)

    def test_round_trip_edge_weights(self, sample_graph, temp_db):
        """Ваги ребер зберігаються точно."""
        net_id = temp_db.save_network("Тест", sample_graph)
        loaded = temp_db.load_network(net_id)

        orig = sample_graph.get_edge(1, 2)
        rest = loaded.get_edge(1, 2)
        assert rest.distance == pytest.approx(orig.distance)
        assert rest.weight == pytest.approx(orig.weight)

    def test_load_nonexistent_returns_none(self, temp_db):
        """Завантаження неіснуючої мережі повертає None."""
        assert temp_db.load_network(9999) is None

    def test_list_networks(self, sample_graph, temp_db):
        """Список мереж відображає збережені."""
        temp_db.save_network("Мережа-1", sample_graph)
        temp_db.save_network("Мережа-2", sample_graph)
        networks = temp_db.list_networks()
        assert len(networks) == 2

    def test_delete_network(self, sample_graph, temp_db):
        """Видалення мережі з БД."""
        net_id = temp_db.save_network("На видалення", sample_graph)
        assert temp_db.delete_network(net_id)
        assert temp_db.load_network(net_id) is None


# БАЗА ДАНИХ SQLite — РЕЗУЛЬТАТИ МКД

class TestDatabaseMSTResults:
    """Тести збереження результатів побудови МКД у БД."""

    def test_save_and_retrieve_mst(self, sample_graph, temp_db):
        """Зберігаємо MSTResult → отримуємо з історії → порівнюємо."""
        net_id = temp_db.save_network("Тест МКД", sample_graph)
        mst_result = PrimMST(sample_graph).find_mst()

        result_id = temp_db.save_mst_result(net_id, mst_result, algorithm="Prim (Binary Heap)")
        assert result_id > 0

        history = temp_db.get_mst_results(net_id)
        assert len(history) == 1
        assert history[0]['algorithm'] == "Prim (Binary Heap)"
        assert history[0]['total_cost'] == pytest.approx(mst_result.total_cost)
        assert history[0]['edge_count'] == mst_result.edge_count

    def test_multiple_algorithms_in_history(self, sample_graph, temp_db):
        """Збереження результатів обох алгоритмів для однієї мережі."""
        net_id = temp_db.save_network("Порівняння", sample_graph)

        prim = PrimMST(sample_graph).find_mst()
        kruskal = KruskalMST(sample_graph).find_mst()

        temp_db.save_mst_result(net_id, prim, algorithm=prim.algorithm_name)
        temp_db.save_mst_result(net_id, kruskal, algorithm=kruskal.algorithm_name)

        history = temp_db.get_mst_results(net_id)
        assert len(history) == 2

        algo_names = {r['algorithm'] for r in history}
        assert "Prim (Binary Heap)" in algo_names
        assert "Kruskal (Union-Find)" in algo_names

    def test_delete_mst_result(self, sample_graph, temp_db):
        """Видалення результату МКД."""
        net_id = temp_db.save_network("Тест", sample_graph)
        mst = PrimMST(sample_graph).find_mst()
        result_id = temp_db.save_mst_result(net_id, mst)

        assert temp_db.delete_mst_result(result_id)
        assert len(temp_db.get_mst_results(net_id)) == 0


# БАЗА ДАНИХ SQLite — БЕНЧМАРКИ


class TestDatabaseBenchmarks:
    """Тести збереження результатів бенчмарків."""

    def test_save_and_get_benchmark(self, temp_db):
        """Зберігаємо бенчмарк → отримуємо → порівнюємо."""
        temp_db.save_benchmark(
            algorithm="Prim (Binary Heap)",
            node_count=100,
            edge_count=4950,
            mean_time_ms=1.23,
            std_time_ms=0.15,
            mst_cost=500000.0
        )
        results = temp_db.get_benchmarks(algorithm="Prim (Binary Heap)")
        assert len(results) == 1
        assert results[0]['node_count'] == 100
        assert results[0]['mean_time_ms'] == pytest.approx(1.23)

    def test_clear_benchmarks(self, temp_db):
        """Очищення всіх бенчмарків."""
        temp_db.save_benchmark("Prim", 10, 45, 0.5, 0.1, 1000)
        temp_db.save_benchmark("Kruskal", 10, 45, 0.6, 0.2, 1000)
        cleared = temp_db.clear_benchmarks()
        assert cleared == 2
        assert len(temp_db.get_benchmarks()) == 0

# БАЗА ДАНИХ SQLite — JSON ІМПОРТ/ЕКСПОРТ


class TestDatabaseJsonExport:
    """Тести експорту/імпорту мережі через JSON-рядок у БД."""

    def test_export_import_round_trip(self, sample_graph, temp_db):
        """Мережа → БД → JSON-рядок → БД (інша мережа) → порівняння."""
        net_id = temp_db.save_network("Оригінал", sample_graph)

        # Експортуємо в JSON
        json_str = temp_db.export_network_json(net_id)
        assert json_str is not None

        # Імпортуємо назад (створюється нова мережа)
        new_id = temp_db.import_network_json(json_str)
        assert new_id != net_id

        # Порівнюємо
        original = temp_db.load_network(net_id)
        imported = temp_db.load_network(new_id)
        assert imported.node_count() == original.node_count()
        assert imported.edge_count() == original.edge_count()


# БАЗА ДАНИХ SQLite — СТАТИСТИКА

class TestDatabaseStats:
    """Тести статистики БД."""

    def test_stats_after_save(self, sample_graph, temp_db):
        """Статистика коректно враховує збережені дані."""
        temp_db.save_network("Тест", sample_graph)
        stats = temp_db.get_stats()
        assert stats['networks'] == 1
        assert stats['nodes'] == 3
        assert stats['edges'] == 2


# ЕКСПОРТ JSON (DataExporter)


class TestJsonExport:
    """Тести експорту результатів МКД у JSON файл."""

    def test_export_creates_file(self, sample_graph, temp_json):
        """Файл створюється."""
        mst = PrimMST(sample_graph).find_mst()
        DataExporter.to_json(mst, sample_graph, temp_json)
        assert os.path.exists(temp_json)

    def test_export_json_structure(self, sample_graph, temp_json):
        """JSON містить обов'язкові секції."""
        mst = PrimMST(sample_graph).find_mst()
        DataExporter.to_json(mst, sample_graph, temp_json)

        with open(temp_json, 'r', encoding='utf-8') as f:
            data = json.load(f)

        assert "metadata" in data
        assert "summary" in data
        assert "mst_edges" in data
        assert "nodes" in data
        assert data["metadata"]["graph_nodes"] == 3

    def test_export_json_cost_match(self, sample_graph, temp_json):
        """Вартість МКД у JSON збігається з MSTResult."""
        mst = PrimMST(sample_graph).find_mst()
        DataExporter.to_json(mst, sample_graph, temp_json)

        with open(temp_json, 'r', encoding='utf-8') as f:
            data = json.load(f)

        assert data["summary"]["total_cost"] == pytest.approx(mst.total_cost)


# ЕКСПОРТ CSV (DataExporter)


class TestCsvExport:
    """Тести експорту ребер МКД у CSV файл."""

    def test_export_creates_file(self, sample_graph, temp_csv):
        """CSV файл створюється."""
        mst = PrimMST(sample_graph).find_mst()
        DataExporter.to_csv(mst, sample_graph, temp_csv)
        assert os.path.exists(temp_csv)

    def test_csv_row_count(self, sample_graph, temp_csv):
        """Кількість рядків = кількість ребер МКД + 1 (заголовок)."""
        mst = PrimMST(sample_graph).find_mst()
        DataExporter.to_csv(mst, sample_graph, temp_csv)

        with open(temp_csv, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)

        # Заголовок + ребра МКД
        assert len(rows) == mst.edge_count + 1


# ЗАВАНТАЖЕННЯ JSON (DataLoader)

class TestDataLoader:
    """Тести завантажувача даних."""

    def test_load_json_basic(self, temp_json):
        """Базове завантаження графа з JSON."""
        json_data = {
            "config": {"default_cost_per_meter": 200.0},
            "nodes": [
                {"id": 10, "name": "Вузол-10", "x": 48.0, "y": 35.0},
                {"id": 20, "name": "Вузол-20", "x": 48.1, "y": 35.1}
            ],
            "edges": [
                {"node1": 10, "node2": 20, "distance": 5000.0}
            ]
        }
        with open(temp_json, 'w', encoding='utf-8') as f:
            json.dump(json_data, f)

        loaded = DataLoader.load_json(temp_json)
        assert loaded.node_count() == 2
        assert loaded.edge_count() == 1

    def test_load_json_default_cost(self, temp_json):
        """cost_per_meter береться з config, якщо не вказано у ребрі."""
        json_data = {
            "config": {"default_cost_per_meter": 200.0},
            "nodes": [
                {"id": 1, "name": "A", "x": 48.0, "y": 35.0},
                {"id": 2, "name": "B", "x": 48.1, "y": 35.1}
            ],
            "edges": [
                {"node1": 1, "node2": 2, "distance": 5000.0}
            ]
        }
        with open(temp_json, 'w', encoding='utf-8') as f:
            json.dump(json_data, f)

        loaded = DataLoader.load_json(temp_json)
        edge = loaded.get_edge(1, 2)
        assert edge.cost_per_meter == 200.0
        assert edge.weight == pytest.approx(5000.0 * 200.0)

    def test_load_json_generates_complete_if_no_edges(self, temp_json):
        """Якщо edges відсутні — генерується повний граф."""
        json_data = {
            "nodes": [
                {"id": 1, "name": "A", "x": 48.0, "y": 35.0},
                {"id": 2, "name": "B", "x": 48.1, "y": 35.1},
                {"id": 3, "name": "C", "x": 48.2, "y": 35.2}
            ]
        }
        with open(temp_json, 'w', encoding='utf-8') as f:
            json.dump(json_data, f)

        loaded = DataLoader.load_json(temp_json)
        assert loaded.edge_count() == 3  # K3 = 3 ребра

    def test_validate_graph_connected(self, sample_graph):
        """Валідація зв'язного графа — без помилок."""
        errors = DataLoader.validate_graph(sample_graph)
        # Граф з 3 вершинами і 2 ребрами (ланцюг) — зв'язний
        assert len(errors) == 0

    def test_validate_graph_disconnected(self):
        """Валідація незв'язного графа — виявляє помилку."""
        g = Graph()
        g.add_node(Node(1, "A", 0.0, 0.0))
        g.add_node(Node(2, "B", 1.0, 0.0))
        g.add_node(Node(3, "C", 5.0, 5.0))
        g.add_edge(Edge(1, 2, distance=100))
        # Вершина 3 ізольована

        errors = DataLoader.validate_graph(g)
        assert any("зв'язним" in e for e in errors)


# ПАКЕТНЕ ТЕСТУВАННЯ

class TestBatchProcessing:
    """
    Імітація реалістичного сценарію: пакетне збереження мереж різних
    розмірів, запуск обох алгоритмів, збереження історії для аналізу.
    """

    def test_batch_save_and_verify(self, temp_db):
        """
        Для кожного розміру графа:
        1. Генеруємо мережу
        2. Зберігаємо в БД
        3. Запускаємо Пріма і Крускала
        4. Зберігаємо обидва результати
        5. Перевіряємо, що вартості збігаються
        """
        gen = GraphGenerator(seed=42)
        sizes = [10, 15, 20]

        for size in sizes:
            graph = gen.random_graph(size, cost_per_meter=150.0, complete=True)
            net_id = temp_db.save_network(f"Batch_V{size}", graph)

            prim_result = PrimMST(graph, record_steps=False).find_mst()
            kruskal_result = KruskalMST(graph, record_steps=False).find_mst()

            temp_db.save_mst_result(net_id, prim_result, algorithm=prim_result.algorithm_name)
            temp_db.save_mst_result(net_id, kruskal_result, algorithm=kruskal_result.algorithm_name)

        # Перевіряємо цілісність
        networks = temp_db.list_networks()
        assert len(networks) == len(sizes)

        for net in networks:
            history = temp_db.get_mst_results(net['id'])
            assert len(history) == 2  # Пріма + Крускал

            costs = [r['total_cost'] for r in history]
            assert costs[0] == pytest.approx(costs[1], rel=1e-5), (
                f"Мережа {net['name']}: Пріма={costs[0]:.2f}, Крускал={costs[1]:.2f}"
            )

    def test_mst_preserves_optimality_after_db_round_trip(self, temp_db):
        """
        Ключовий тест: МКД, побудоване на завантаженому з БД графі,
        має таку ж вартість, як МКД на оригіналі.
        Доводить, що БД не вносить похибку, яка б змінила МКД.
        """
        gen = GraphGenerator(seed=42)
        original_graph = gen.random_graph(15, cost_per_meter=150.0, complete=True)
        original_cost = PrimMST(original_graph, record_steps=False).find_mst().total_cost

        # Зберігаємо та завантажуємо
        net_id = temp_db.save_network("Round-trip тест", original_graph)
        loaded_graph = temp_db.load_network(net_id)
        loaded_cost = PrimMST(loaded_graph, record_steps=False).find_mst().total_cost

        assert loaded_cost == pytest.approx(original_cost, rel=1e-5)