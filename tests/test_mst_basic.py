"""
test_mst_basic.py — Тести коректності алгоритмів МКД

Рівень тестування: Unit (одиничне)
Мета: Перевірити, що алгоритми Пріма та Крускала правильно будують
      мінімальне кістякове дерево для різних топологій графів,
      включаючи граничні та вироджені випадки.

Математичні інваріанти:
    1. |E(МКД)| = V - 1 для зв'язного графа з V вершинами
    2. МКД — ациклічне (є деревом)
    3. Вартість МКД ≤ вартості будь-якого іншого кістякового дерева
    4. Якщо всі ваги ребер різні — МКД єдине (Теорема 2)

Граничні випадки:
    - Порожній граф (0 вершин)
    - Тривіальний граф (1 вершина)
    - Мінімальний граф (2 вершини)
    - Незв'язний граф (кістяковий ліс)
    - Рівні ваги (неоднозначність МКД)
    - Топологія «Зірка»
    - Топологія «Ланцюг» (єдине кістякове дерево)
"""

import pytest
import warnings
from models import Graph, Node, Edge
from algorithms import PrimMST, KruskalMST


# ДОПОМІЖНІ ФУНКЦІЇ

def assert_mst_invariants(result, graph, label=""):
    """
    Перевіряє математичні інваріанти МКД (з підрозділу 1.2).

    Args:
        result: MSTResult від алгоритму
        graph: вхідний Graph
        label: мітка для повідомлень про помилку
    """
    V = graph.node_count()

    # Інваріант 1: |E(МКД)| = V - 1 (для зв'язного графа)
    if V > 0 and graph.is_connected():
        assert len(result.edges) == V - 1, (
            f"{label}: МКД має містити {V - 1} ребер, отримано {len(result.edges)}"
        )

    # Інваріант 2: МКД — ациклічне (ребра не утворюють циклів)
    if result.edges:
        adj = {}
        for e in result.edges:
            adj.setdefault(e.node1, []).append(e.node2)
            adj.setdefault(e.node2, []).append(e.node1)
        # Для дерева: |E| = |V_mst| - 1 (де V_mst — вершини, покриті МКД)
        mst_vertices = set()
        for e in result.edges:
            mst_vertices.add(e.node1)
            mst_vertices.add(e.node2)
        assert len(result.edges) == len(mst_vertices) - 1, (
            f"{label}: МКД має цикл! Ребер: {len(result.edges)}, "
            f"Вершин: {len(mst_vertices)}"
        )

    # Інваріант 3: total_cost == сума ваг ребер
    expected_cost = sum(e.weight for e in result.edges)
    assert result.total_cost == pytest.approx(expected_cost), (
        f"{label}: total_cost ({result.total_cost}) != сума ваг ({expected_cost})"
    )

    # Інваріант 4: total_distance == сума довжин ребер
    expected_dist = sum(e.distance for e in result.edges)
    assert result.total_distance == pytest.approx(expected_dist), (
        f"{label}: total_distance ({result.total_distance}) != сума довжин ({expected_dist})"
    )


# ГРАНИЧНІ ВИПАДКИ

class TestEmptyGraph:
    """Тест 1: Порожній граф (0 вершин)."""

    def test_prim_empty(self):
        """Пріма на порожньому графі повертає порожнє МКД без помилок."""
        g = Graph()
        result = PrimMST(g).find_mst()
        assert result.total_cost == 0.0
        assert len(result.edges) == 0

    def test_kruskal_empty(self):
        """Крускал на порожньому графі повертає порожнє МКД без помилок."""
        g = Graph()
        result = KruskalMST(g).find_mst()
        assert result.total_cost == 0.0
        assert len(result.edges) == 0


class TestSingleNode:
    """Тест 2: Граф з однією вершиною — МКД не має ребер."""

    def test_prim_single(self):
        g = Graph()
        g.add_node(Node(1, "Одинока ПС", 48.0, 35.0))
        result = PrimMST(g).find_mst()
        assert result.total_cost == 0.0
        assert len(result.edges) == 0

    def test_kruskal_single(self):
        g = Graph()
        g.add_node(Node(1, "Одинока ПС", 48.0, 35.0))
        result = KruskalMST(g).find_mst()
        assert result.total_cost == 0.0
        assert len(result.edges) == 0


class TestTwoNodes:
    """Тест 3: Граф з двома вершинами — МКД = єдине ребро."""

    def test_prim_two_nodes(self):
        g = Graph()
        g.add_node(Node(1, "A", 48.0, 35.0))
        g.add_node(Node(2, "B", 48.1, 35.1))
        g.add_edge(Edge(1, 2, distance=1000.0, cost_per_meter=150.0))

        result = PrimMST(g).find_mst()
        assert len(result.edges) == 1
        assert result.total_cost == pytest.approx(150_000.0)
        assert_mst_invariants(result, g, "Пріма K2")

    def test_kruskal_two_nodes(self):
        g = Graph()
        g.add_node(Node(1, "A", 48.0, 35.0))
        g.add_node(Node(2, "B", 48.1, 35.1))
        g.add_edge(Edge(1, 2, distance=1000.0, cost_per_meter=150.0))

        result = KruskalMST(g).find_mst()
        assert len(result.edges) == 1
        assert result.total_cost == pytest.approx(150_000.0)
        assert_mst_invariants(result, g, "Крускал K2")


class TestDisconnectedGraph:
    """
    Тест 4: Незв'язний граф (2 окремі компоненти).
    Алгоритми мають побудувати кістяковий ліс і видати RuntimeWarning,
    як передбачено в base_mst.py (рядок 154).
    """

    @pytest.fixture
    def disconnected_graph(self):
        g = Graph()
        g.add_node(Node(1, "A", 0.0, 0.0))
        g.add_node(Node(2, "B", 1.0, 0.0))
        g.add_node(Node(3, "C", 5.0, 5.0))
        g.add_node(Node(4, "D", 6.0, 5.0))
        g.add_edge(Edge(1, 2, distance=10, cost_per_meter=100))   # 1000 грн
        g.add_edge(Edge(3, 4, distance=15, cost_per_meter=100))   # 1500 грн
        return g

    def test_prim_warns(self, disconnected_graph):
        """Пріма видає попередження про незв'язність."""
        with pytest.warns(RuntimeWarning, match="незв'язний"):
            result = PrimMST(disconnected_graph).find_mst()
        # Кістяковий ліс: 2 компоненти × 1 ребро = 2 ребра (не 3 = V-1)
        assert len(result.edges) == 2
        assert result.total_cost == pytest.approx(2500.0)

    def test_kruskal_warns(self, disconnected_graph):
        """Крускал видає попередження про незв'язність."""
        with pytest.warns(RuntimeWarning, match="незв'язний"):
            result = KruskalMST(disconnected_graph).find_mst()
        assert len(result.edges) == 2
        assert result.total_cost == pytest.approx(2500.0)


class TestEqualWeights:
    """
    Тест 5: Граф з однаковими вагами ребер (квадрат).
    Пріма і Крускал можуть обрати РІЗНІ ребра (неоднозначність МКД,
    коли ваги не є попарно різними — див. Теорему 2 підрозділу 1.2),
    але сумарна вартість має бути однаковою.
    """

    def test_equal_weights_cost_match(self):
        g = Graph()
        for i in range(1, 5):
            g.add_node(Node(i, f"N{i}", 0.0, float(i)))
        # Квадрат: 1-2-3-4-1, всі ваги = 1000
        for n1, n2 in [(1, 2), (2, 3), (3, 4), (4, 1)]:
            g.add_edge(Edge(n1, n2, distance=10, cost_per_meter=100))

        prim = PrimMST(g).find_mst()
        kruskal = KruskalMST(g).find_mst()

        # МКД для 4 вершин: 3 ребра, кожне вагою 1000 → всього 3000
        assert len(prim.edges) == 3
        assert prim.total_cost == pytest.approx(3000.0)
        assert kruskal.total_cost == pytest.approx(prim.total_cost)
        assert_mst_invariants(prim, g, "рівні ваги")

# КЛАСИЧНІ ТОПОЛОГІЇ


class TestStarTopology:
    """
    Тест 6: Топологія «Зірка».
    Центральний вузол з'єднаний з усіма іншими.
    Єдиний можливий кістяк = всі промені (бо інших ребер немає).
    """

    def test_star_prim(self):
        g = Graph()
        g.add_node(Node(0, "Центр", 0.0, 0.0))

        total = 0
        for i in range(1, 6):
            g.add_node(Node(i, f"Промінь-{i}", float(i), float(i)))
            g.add_edge(Edge(0, i, distance=float(i), cost_per_meter=100))
            total += i * 100  # 100 + 200 + 300 + 400 + 500 = 1500

        result = PrimMST(g).find_mst()
        assert len(result.edges) == 5
        assert result.total_cost == pytest.approx(float(total))
        assert_mst_invariants(result, g, "зірка")


class TestChainTopology:
    """
    Тест 7: Топологія «Ланцюг» (1-2-3-4-5).
    Тільки один кістяковий граф — сам ланцюг.
    МКД обов'язково = весь ланцюг.
    """

    def test_chain(self):
        g = Graph()
        for i in range(5):
            g.add_node(Node(i, f"N{i}", 0.0, float(i)))
        for i in range(4):
            g.add_edge(Edge(i, i + 1, distance=100, cost_per_meter=150))

        prim = PrimMST(g).find_mst()
        kruskal = KruskalMST(g).find_mst()

        assert len(prim.edges) == 4
        assert prim.total_cost == pytest.approx(4 * 100 * 150)
        assert kruskal.total_cost == pytest.approx(prim.total_cost)
        assert_mst_invariants(prim, g, "ланцюг Пріма")
        assert_mst_invariants(kruskal, g, "ланцюг Крускала")


class TestHandCalculatedMST:
    """
    Тест 8: Граф з ручним розрахунком оптимального МКД.

    Топологія (повний граф K4):
        1 --100-- 2
        |  \    / |
       300  250 150
        |  /    \ |
        3 --200-- 4

    Ваги: (1,2)=100, (2,4)=150, (3,4)=200, (1,3)=300, (1,4)=250, (2,3)=350
    Сортування: 100, 150, 200, 250, 300, 350
    Крускал бере: (1,2)=100, (2,4)=150, (3,4)=200 → всього 450
    """

    @pytest.fixture
    def k4_graph(self):
        g = Graph()
        for i in range(1, 5):
            g.add_node(Node(i, f"N{i}", 0.0, float(i)))
        g.add_edge(Edge(1, 2, distance=1.0, cost_per_meter=100))   # 100
        g.add_edge(Edge(2, 4, distance=1.0, cost_per_meter=150))   # 150
        g.add_edge(Edge(3, 4, distance=1.0, cost_per_meter=200))   # 200
        g.add_edge(Edge(1, 4, distance=1.0, cost_per_meter=250))   # 250
        g.add_edge(Edge(1, 3, distance=1.0, cost_per_meter=300))   # 300
        g.add_edge(Edge(2, 3, distance=1.0, cost_per_meter=350))   # 350
        return g

    def test_prim_optimal(self, k4_graph):
        result = PrimMST(k4_graph).find_mst()
        assert result.total_cost == pytest.approx(450.0)
        assert len(result.edges) == 3
        assert_mst_invariants(result, k4_graph, "K4 Пріма")

    def test_kruskal_optimal(self, k4_graph):
        result = KruskalMST(k4_graph).find_mst()
        assert result.total_cost == pytest.approx(450.0)
        assert len(result.edges) == 3
        assert_mst_invariants(result, k4_graph, "K4 Крускала")


# МАТЕМАТИЧНІ ІНВАРІАНТИ

class TestMSTInvariants:
    """
    Перевірка фундаментальних властивостей МКД з підрозділу 1.2 курсової.
    Тести виконуються на графах різних розмірів.
    """

    @pytest.mark.parametrize("n", [3, 5, 10, 20])
    def test_edge_count_equals_v_minus_1(self, n):
        """
        Інваріант: |E(МКД)| = V - 1.
        Фундаментальна властивість дерева: воно з'єднує V вершин
        рівно V-1 ребрами без циклів.
        """
        from data.generator import GraphGenerator
        gen = GraphGenerator(seed=42)
        g = gen.random_graph(num_nodes=n, cost_per_meter=150.0, complete=True)

        prim = PrimMST(g).find_mst()
        assert len(prim.edges) == n - 1

    @pytest.mark.parametrize("n", [5, 10, 15])
    def test_mst_connects_all_vertices(self, n):
        """
        Інваріант: МКД покриває всі вершини графа.
        Кожна підстанція має бути підключена.
        """
        from data.generator import GraphGenerator
        gen = GraphGenerator(seed=7)
        g = gen.random_graph(num_nodes=n, cost_per_meter=150.0, complete=True)

        result = PrimMST(g).find_mst()
        mst_vertices = set()
        for e in result.edges:
            mst_vertices.add(e.node1)
            mst_vertices.add(e.node2)
        assert mst_vertices == set(g.get_node_ids())

    def test_mst_cost_less_than_full_graph(self):
        """
        Вартість МКД < загальної вартості всіх ребер графа
        (для графа з більш ніж V-1 ребрами).
        """
        from data.generator import GraphGenerator
        gen = GraphGenerator(seed=42)
        g = gen.random_graph(num_nodes=10, cost_per_meter=150.0, complete=True)

        result = PrimMST(g).find_mst()
        total_graph_cost = sum(e.weight for e in g.iter_edges())

        assert result.total_cost < total_graph_cost


# ПОКРОКОВА ВІЗУАЛІЗАЦІЯ (record_steps)

class TestRecordSteps:
    """
    Тести покрокової візуалізації — кожен крок алгоритму зберігається
    для подальшого відтворення в анімації (вкладка «Анімація» у Dash).
    """

    def test_steps_count_equals_edges(self):
        """Кількість кроків = кількість ребер МКД (кожен крок додає 1 ребро)."""
        from data.generator import GraphGenerator
        gen = GraphGenerator(seed=42)
        g = gen.random_graph(num_nodes=8, cost_per_meter=150.0, complete=True)

        result = PrimMST(g, record_steps=True).find_mst()
        assert len(result.steps) == len(result.edges)

    def test_steps_grow_monotonically(self):
        """Кожен наступний крок містить на 1 ребро більше (монотонне зростання)."""
        from data.generator import GraphGenerator
        gen = GraphGenerator(seed=42)
        g = gen.random_graph(num_nodes=6, cost_per_meter=150.0, complete=True)

        result = PrimMST(g, record_steps=True).find_mst()
        for i in range(len(result.steps)):
            assert len(result.steps[i]) == i + 1

    def test_no_steps_when_disabled(self):
        """При record_steps=False кроки не зберігаються (для бенчмарків)."""
        from data.generator import GraphGenerator
        gen = GraphGenerator(seed=42)
        g = gen.random_graph(num_nodes=10, cost_per_meter=150.0, complete=True)

        result = PrimMST(g, record_steps=False).find_mst()
        assert len(result.steps) == 0
        # Але МКД все одно побудовано коректно
        assert len(result.edges) == 9


# МЕТАДАНІ РЕЗУЛЬТАТУ


class TestMSTResultMetadata:
    """Перевірка правильності заповнення MSTResult."""

    def test_algorithm_name_prim(self):
        """Пріма повертає свою назву."""
        g = Graph()
        g.add_node(Node(1, "A", 0.0, 0.0))
        g.add_node(Node(2, "B", 1.0, 1.0))
        g.add_edge(Edge(1, 2, distance=100))
        result = PrimMST(g).find_mst()
        assert "Prim" in result.algorithm_name

    def test_algorithm_name_kruskal(self):
        """Крускал повертає свою назву."""
        g = Graph()
        g.add_node(Node(1, "A", 0.0, 0.0))
        g.add_node(Node(2, "B", 1.0, 1.0))
        g.add_edge(Edge(1, 2, distance=100))
        result = KruskalMST(g).find_mst()
        assert "Kruskal" in result.algorithm_name

    def test_execution_time_positive(self):
        """Час виконання > 0 (навіть для маленьких графів)."""
        from data.generator import GraphGenerator
        gen = GraphGenerator(seed=42)
        g = gen.random_graph(num_nodes=20, cost_per_meter=150.0, complete=True)

        result = PrimMST(g).find_mst()
        assert result.execution_time > 0

    def test_result_to_dict(self):
        """MSTResult.to_dict() містить всі необхідні поля."""
        g = Graph()
        g.add_node(Node(1, "A", 0.0, 0.0))
        g.add_node(Node(2, "B", 1.0, 1.0))
        g.add_edge(Edge(1, 2, distance=100, cost_per_meter=200))

        result = PrimMST(g).find_mst()
        d = result.to_dict()
        assert 'algorithm' in d
        assert 'total_cost' in d
        assert 'total_distance_m' in d
        assert 'num_edges' in d
        assert 'edges' in d
        assert d['total_cost'] == pytest.approx(20000.0)