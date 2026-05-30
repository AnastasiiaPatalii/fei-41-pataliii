"""
test_models.py — Модульні тести структур даних (Node, Edge, Graph)

Рівень тестування: Unit (одиничне)
Мета: Довести коректність фундаментальних структур даних, на яких
      базується вся система. Якщо модель зламана — алгоритми видадуть
      неправильний результат, але їхні тести цього не виявлять.

Покриття:
    - Node: валідація, distance_to (гаверсинус, евклід), серіалізація
    - Edge: валідація, weight, нормалізація, порівняння, серіалізація
    - Graph: додавання/видалення вершин і ребер, зв'язність,
             серіалізація (to_dict / from_dict), генерація повного графа,
             фабричні методи, оператор __contains__
"""

import pytest
import math
from models import Node, Edge, Graph
from models.graph import NodeNotFoundError, EdgeNotFoundError


# ТЕСТИ NODE — Підстанція (вершина графа)

class TestNodeCreation:
    """Тести створення та валідації вершин."""

    def test_valid_node(self):
        """Створення коректної вершини з усіма параметрами."""
        node = Node(1, "ПС Центральна", 48.4647, 35.0462, metadata={"type": "main"})
        assert node.id == 1
        assert node.name == "ПС Центральна"
        assert node.x == 48.4647
        assert node.y == 35.0462
        assert node.metadata == {"type": "main"}

    def test_negative_id_raises(self):
        """ID підстанції не може бути від'ємним."""
        with pytest.raises(ValueError, match="невід'ємним"):
            Node(-1, "Помилка", 48.0, 35.0)

    def test_empty_name_raises(self):
        """Назва підстанції не може бути порожньою."""
        with pytest.raises(ValueError, match="порожньою"):
            Node(1, "   ", 48.0, 35.0)

    def test_latitude_out_of_range(self):
        """Широта обмежена діапазоном [-90, 90]."""
        with pytest.raises(ValueError, match="Широта"):
            Node(1, "Помилка", 91.0, 35.0)

    def test_longitude_out_of_range(self):
        """Довгота обмежена діапазоном [-180, 180]."""
        with pytest.raises(ValueError, match="Довгота"):
            Node(1, "Помилка", 48.0, 181.0)

    def test_default_metadata_is_empty_dict(self):
        """За замовчуванням metadata — порожній словник."""
        node = Node(0, "Тест", 0.0, 0.0)
        assert node.metadata == {}


class TestNodeDistance:
    """
    Тести обчислення відстані між підстанціями.

    Критично важлива частина: від коректності distance_to() залежить
    вага КОЖНОГО ребра в графі, а отже і результат МКД.
    """

    def test_haversine_known_distance(self):
        """
        Еталонний тест: Київ → Полтава ≈ 300 км (по прямій).
        Допуск ±5 км враховує різницю між сферичною та еліпсоїдальною моделлю.
        """
        kyiv = Node(1, "Київ", 50.4501, 30.5234)
        poltava = Node(2, "Полтава", 49.5883, 34.5514)
        distance_m = kyiv.distance_to(poltava, method='haversine')
        distance_km = distance_m / 1000
        assert distance_km == pytest.approx(300, abs=5)

    def test_haversine_same_point_is_zero(self):
        """Відстань від точки до самої себе дорівнює нулю."""
        node = Node(1, "Тест", 48.5, 35.0)
        assert node.distance_to(node, method='haversine') == pytest.approx(0.0, abs=1e-6)

    def test_haversine_symmetry(self):
        """Гаверсинус симетричний: d(A, B) == d(B, A)."""
        a = Node(1, "A", 48.0, 35.0)
        b = Node(2, "B", 49.0, 36.0)
        assert a.distance_to(b) == pytest.approx(b.distance_to(a))

    def test_haversine_triangle_inequality(self):
        """Нерівність трикутника: d(A, C) ≤ d(A, B) + d(B, C)."""
        a = Node(1, "A", 48.0, 34.0)
        b = Node(2, "B", 48.5, 35.0)
        c = Node(3, "C", 49.0, 36.0)
        assert a.distance_to(c) <= a.distance_to(b) + b.distance_to(c) + 1e-6

    def test_euclidean_distance(self):
        """Евклідова відстань для тестових координат (трикутник 3-4-5)."""
        a = Node(1, "A", 0.0, 0.0)
        b = Node(2, "B", 3.0, 4.0)
        assert a.distance_to(b, method='euclidean') == pytest.approx(5.0)

    def test_unknown_method_raises(self):
        """Невідомий метод обчислення має викликати ValueError."""
        a = Node(1, "A", 0.0, 0.0)
        b = Node(2, "B", 1.0, 1.0)
        with pytest.raises(ValueError, match="Невідомий метод"):
            a.distance_to(b, method='manhattan')

    def test_haversine_returns_positive(self):
        """Відстань завжди додатна для різних точок."""
        a = Node(1, "A", 48.0, 35.0)
        b = Node(2, "B", 49.0, 36.0)
        assert a.distance_to(b) > 0


class TestNodeSerialization:
    """Тести серіалізації / десеріалізації Node."""

    def test_to_dict_and_back(self):
        """Round-trip: Node → dict → Node зберігає всі поля."""
        original = Node(5, "ПС-5", 48.123, 35.456, metadata={"power": 100})
        restored = Node.from_dict(original.to_dict())
        assert restored.id == original.id
        assert restored.name == original.name
        assert restored.x == pytest.approx(original.x)
        assert restored.y == pytest.approx(original.y)
        assert restored.metadata == original.metadata

    def test_equality_by_id(self):
        """Два вузли з однаковим ID вважаються рівними."""
        n1 = Node(1, "Перша", 48.0, 35.0)
        n2 = Node(1, "Друга", 49.0, 36.0)
        assert n1 == n2

    def test_hash_by_id(self):
        """Хеш залежить тільки від ID — для використання у set та dict."""
        n1 = Node(1, "Перша", 48.0, 35.0)
        n2 = Node(1, "Друга", 49.0, 36.0)
        assert hash(n1) == hash(n2)
        assert len({n1, n2}) == 1

# ТЕСТИ EDGE — Лінія електропередачі (ребро графа)

class TestEdgeCreation:
    """Тести створення та валідації ребер."""

    def test_valid_edge(self):
        """Створення коректного ребра з обчисленням ваги."""
        e = Edge(1, 2, distance=10000.0, cost_per_meter=150.0)
        assert e.weight == 10000.0 * 150.0  # 1 500 000 грн

    def test_self_loop_raises(self):
        """Петля (ребро в саму себе) заборонена."""
        with pytest.raises(ValueError, match="Петля"):
            Edge(1, 1, distance=100)

    def test_zero_distance_raises(self):
        """Довжина лінії має бути додатною."""
        with pytest.raises(ValueError, match="додатною"):
            Edge(1, 2, distance=0)

    def test_negative_cost_raises(self):
        """Вартість за метр не може бути від'ємною."""
        with pytest.raises(ValueError, match="від'ємною"):
            Edge(1, 2, distance=100, cost_per_meter=-10)

    def test_node_normalization(self):
        """Ребро (5, 3) автоматично нормалізується до (3, 5)."""
        e = Edge(5, 3, distance=100)
        assert e.node1 == 3
        assert e.node2 == 5

    def test_nodes_property(self):
        """Властивість nodes повертає відсортований кортеж."""
        e = Edge(10, 3, distance=100)
        assert e.nodes == (3, 10)


class TestEdgeWeight:
    """Тести обчислення ваги ребра (ключова формула для МКД)."""

    def test_weight_formula(self):
        """Вага = distance × cost_per_meter."""
        e = Edge(1, 2, distance=5000.0, cost_per_meter=120.0)
        assert e.weight == pytest.approx(600_000.0)

    def test_default_cost(self):
        """За замовчуванням cost_per_meter = 100.0."""
        e = Edge(1, 2, distance=1000.0)
        assert e.weight == pytest.approx(100_000.0)

    def test_ordering_by_weight(self):
        """Оператор < порівнює за вагою (потрібен для heapq у Прімі)."""
        light = Edge(1, 2, distance=100, cost_per_meter=10)     # 1 000
        heavy = Edge(3, 4, distance=100, cost_per_meter=100)    # 10 000
        assert light < heavy
        assert not heavy < light

    def test_contains_node(self):
        """Метод contains_node перевіряє належність вершини до ребра."""
        e = Edge(3, 7, distance=100)
        assert e.contains_node(3)
        assert e.contains_node(7)
        assert not e.contains_node(5)

    def test_get_other_node(self):
        """Метод get_other_node повертає протилежний кінець."""
        e = Edge(3, 7, distance=100)
        assert e.get_other_node(3) == 7
        assert e.get_other_node(7) == 3


class TestEdgeSerialization:
    """Тести серіалізації ребер."""

    def test_to_dict_and_back(self):
        """Round-trip: Edge → dict → Edge зберігає distance та cost_per_meter."""
        original = Edge(1, 2, distance=12345.67, cost_per_meter=200.0)
        restored = Edge.from_dict(original.to_dict())
        assert restored.node1 == original.node1
        assert restored.node2 == original.node2
        assert restored.distance == pytest.approx(original.distance)
        assert restored.cost_per_meter == pytest.approx(original.cost_per_meter)
        assert restored.weight == pytest.approx(original.weight)

    def test_from_nodes(self):
        """Фабричний метод Edge.from_nodes автоматично обчислює відстань."""
        n1 = Node(1, "A", 48.0, 35.0)
        n2 = Node(2, "B", 48.1, 35.1)
        e = Edge.from_nodes(n1, n2, cost_per_meter=150.0)
        assert e.node1 == 1
        assert e.node2 == 2
        assert e.distance > 0
        assert e.cost_per_meter == 150.0

    def test_edge_equality_by_endpoints(self):
        """Два ребра рівні, якщо з'єднують одні й ті самі вершини."""
        e1 = Edge(1, 2, distance=100)
        e2 = Edge(1, 2, distance=999)  # інша відстань, але ті ж вершини
        assert e1 == e2


# ТЕСТИ GRAPH — Граф енергомережі

@pytest.fixture
def empty_graph():
    """Порожній граф."""
    return Graph()


@pytest.fixture
def triangle_graph():
    """
    Трикутник: 3 вершини, 3 ребра, відомі ваги.

         1
        / \
    100   200
      /     \
     2 --300-- 3

    Ваги: (1,2)=100, (1,3)=200, (2,3)=300
    МКД має містити ребра (1,2) та (1,3), вартість = 300.
    """
    g = Graph()
    g.add_node(Node(1, "A", 0.0, 0.0))
    g.add_node(Node(2, "B", 1.0, 0.0))
    g.add_node(Node(3, "C", 0.0, 1.0))
    g.add_edge(Edge(1, 2, distance=1.0, cost_per_meter=100.0))   # weight=100
    g.add_edge(Edge(1, 3, distance=1.0, cost_per_meter=200.0))   # weight=200
    g.add_edge(Edge(2, 3, distance=1.0, cost_per_meter=300.0))   # weight=300
    return g


class TestGraphNodes:
    """Тести операцій з вершинами."""

    def test_add_and_count(self, empty_graph):
        """Додавання вершин збільшує лічильник."""
        g = empty_graph
        g.add_node(Node(1, "A", 0.0, 0.0))
        g.add_node(Node(2, "B", 1.0, 1.0))
        assert g.node_count() == 2

    def test_add_duplicate_raises(self, empty_graph):
        """Додавання вершини з існуючим ID — помилка."""
        g = empty_graph
        g.add_node(Node(1, "A", 0.0, 0.0))
        with pytest.raises(ValueError, match="вже існує"):
            g.add_node(Node(1, "Дубль", 1.0, 1.0))

    def test_remove_node(self, triangle_graph):
        """Видалення вершини також видаляє всі інцидентні ребра."""
        g = triangle_graph
        g.remove_node(1)
        assert g.node_count() == 2
        # Ребра (1,2) та (1,3) видалені, залишилось тільки (2,3)
        assert g.edge_count() == 1
        assert g.has_edge(2, 3)

    def test_remove_nonexistent_raises(self, empty_graph):
        """Видалення неіснуючої вершини — помилка."""
        with pytest.raises(NodeNotFoundError):
            empty_graph.remove_node(999)

    def test_get_node(self, triangle_graph):
        """get_node повертає об'єкт Node за ID."""
        node = triangle_graph.get_node(1)
        assert node.name == "A"

    def test_has_node(self, triangle_graph):
        """has_node перевіряє наявність."""
        assert triangle_graph.has_node(1)
        assert not triangle_graph.has_node(99)

    def test_is_empty(self, empty_graph):
        """Порожній граф."""
        assert empty_graph.is_empty()

    def test_not_empty(self, triangle_graph):
        """Непорожній граф."""
        assert not triangle_graph.is_empty()


class TestGraphEdges:
    """Тести операцій з ребрами."""

    def test_add_edge_and_count(self, triangle_graph):
        """Трикутний граф має 3 ребра."""
        assert triangle_graph.edge_count() == 3

    def test_add_edge_to_nonexistent_node(self, empty_graph):
        """Ребро між неіснуючими вершинами — помилка."""
        with pytest.raises(NodeNotFoundError):
            empty_graph.add_edge(Edge(1, 2, distance=100))

    def test_add_duplicate_edge_raises(self, triangle_graph):
        """Повторне додавання того ж ребра — помилка."""
        with pytest.raises(ValueError, match="вже існує"):
            triangle_graph.add_edge(Edge(1, 2, distance=999))

    def test_remove_edge(self, triangle_graph):
        """Видалення ребра зменшує лічильник."""
        triangle_graph.remove_edge(1, 2)
        assert triangle_graph.edge_count() == 2
        assert not triangle_graph.has_edge(1, 2)

    def test_remove_edge_reverse_order(self, triangle_graph):
        """Ребро (2, 1) і (1, 2) — одне й те саме (неорієнтований граф)."""
        triangle_graph.remove_edge(2, 1)
        assert not triangle_graph.has_edge(1, 2)

    def test_remove_nonexistent_edge_raises(self, triangle_graph):
        """Видалення неіснуючого ребра — помилка."""
        with pytest.raises(EdgeNotFoundError):
            triangle_graph.remove_edge(1, 99)

    def test_get_edge(self, triangle_graph):
        """get_edge повертає об'єкт Edge."""
        edge = triangle_graph.get_edge(1, 2)
        assert edge.weight == pytest.approx(100.0)

    def test_get_neighbors(self, triangle_graph):
        """get_neighbors повертає список інцидентних ребер."""
        neighbors = triangle_graph.get_neighbors(1)
        assert len(neighbors) == 2  # ребра до 2 і до 3

    def test_get_degree(self, triangle_graph):
        """Степінь вершини = кількість інцидентних ребер."""
        assert triangle_graph.get_degree(1) == 2
        assert triangle_graph.get_degree(2) == 2


class TestGraphConnectivity:
    """Тести аналізу зв'язності."""

    def test_connected_graph(self, triangle_graph):
        """Трикутник — зв'язний граф."""
        assert triangle_graph.is_connected()

    def test_disconnected_graph(self):
        """Граф з двома окремими компонентами — незв'язний."""
        g = Graph()
        g.add_node(Node(1, "A", 0.0, 0.0))
        g.add_node(Node(2, "B", 1.0, 0.0))
        g.add_node(Node(3, "C", 5.0, 5.0))
        g.add_node(Node(4, "D", 6.0, 5.0))
        g.add_edge(Edge(1, 2, distance=100))
        g.add_edge(Edge(3, 4, distance=100))

        assert not g.is_connected()

    def test_connected_components(self):
        """Пошук компонент зв'язності."""
        g = Graph()
        for i in range(4):
            g.add_node(Node(i, f"N{i}", float(i), 0.0))
        g.add_edge(Edge(0, 1, distance=100))
        g.add_edge(Edge(2, 3, distance=100))

        components = g.get_connected_components()
        assert len(components) == 2

    def test_single_node_is_connected(self):
        """Граф з 1 вершиною вважається зв'язним."""
        g = Graph()
        g.add_node(Node(1, "Один", 0.0, 0.0))
        assert g.is_connected()

    def test_empty_graph_is_connected(self, empty_graph):
        """Порожній граф вважається зв'язним (за визначенням)."""
        assert empty_graph.is_connected()


class TestGraphSerialization:
    """Тести серіалізації графа (to_dict / from_dict) — критично для БД та JSON."""

    def test_round_trip(self, triangle_graph):
        """Round-trip: Graph → dict → Graph зберігає структуру."""
        data = triangle_graph.to_dict()
        restored = Graph.from_dict(data)

        assert restored.node_count() == triangle_graph.node_count()
        assert restored.edge_count() == triangle_graph.edge_count()

    def test_round_trip_preserves_weights(self, triangle_graph):
        """Round-trip зберігає ваги ребер."""
        data = triangle_graph.to_dict()
        restored = Graph.from_dict(data)

        original_edge = triangle_graph.get_edge(1, 2)
        restored_edge = restored.get_edge(1, 2)
        assert restored_edge.weight == pytest.approx(original_edge.weight)

    def test_round_trip_preserves_coordinates(self, triangle_graph):
        """Round-trip зберігає координати вершин."""
        data = triangle_graph.to_dict()
        restored = Graph.from_dict(data)

        orig = triangle_graph.get_node(1)
        rest = restored.get_node(1)
        assert rest.x == pytest.approx(orig.x)
        assert rest.y == pytest.approx(orig.y)

    def test_json_file_round_trip(self, triangle_graph, tmp_path):
        """Збереження та завантаження з JSON файлу."""
        path = str(tmp_path / "test_graph.json")
        triangle_graph.to_json(path)
        restored = Graph.from_json(path)
        assert restored.node_count() == 3
        assert restored.edge_count() == 3


class TestGraphGeneration:
    """Тести генерації повного графа."""

    def test_complete_graph_edge_count(self):
        """Повний граф K_n має n(n-1)/2 ребер."""
        g = Graph()
        n = 5
        for i in range(n):
            g.add_node(Node(i, f"N{i}", 48.0 + i * 0.01, 35.0))
        g.generate_complete_graph(cost_per_meter=150.0)

        expected = n * (n - 1) // 2
        assert g.edge_count() == expected

    def test_complete_graph_is_connected(self):
        """Повний граф завжди зв'язний."""
        g = Graph()
        for i in range(4):
            g.add_node(Node(i, f"N{i}", 48.0 + i * 0.01, 35.0))
        g.generate_complete_graph()
        assert g.is_connected()

    def test_from_nodes_factory(self):
        """Фабричний метод from_nodes створює повний граф."""
        nodes = [Node(i, f"N{i}", 48.0 + i * 0.01, 35.0) for i in range(3)]
        g = Graph.from_nodes(nodes, cost_per_meter=100.0)
        assert g.node_count() == 3
        assert g.edge_count() == 3

    def test_from_nodes_and_edges_factory(self):
        """Фабричний метод from_nodes_and_edges."""
        nodes = [Node(1, "A", 0.0, 0.0), Node(2, "B", 1.0, 1.0)]
        edges = [Edge(1, 2, distance=100.0)]
        g = Graph.from_nodes_and_edges(nodes, edges)
        assert g.node_count() == 2
        assert g.edge_count() == 1


class TestGraphContains:
    """Тести оператора in (__contains__)."""

    def test_contains_node_by_id(self, triangle_graph):
        assert 1 in triangle_graph
        assert 99 not in triangle_graph

    def test_contains_edge_by_tuple(self, triangle_graph):
        assert (1, 2) in triangle_graph
        assert (2, 1) in triangle_graph  # неорієнтований
        assert (1, 99) not in triangle_graph


class TestGraphStats:
    """Тести статистики графа."""

    def test_density_complete_graph(self):
        """Щільність повного графа = 1.0."""
        g = Graph()
        for i in range(4):
            g.add_node(Node(i, f"N{i}", 48.0 + i * 0.01, 35.0))
        g.generate_complete_graph()
        stats = g.get_stats()
        assert stats.density == pytest.approx(1.0)

    def test_average_degree(self, triangle_graph):
        """Середній степінь трикутника = 2*E/V = 2*3/3 = 2."""
        stats = triangle_graph.get_stats()
        assert stats.avg_degree == pytest.approx(2.0)