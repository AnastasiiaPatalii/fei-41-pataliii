"""Тести 2-edge-connectivity: пошук мостів (Тар'ян) + доповнення."""

import pytest

from models import Graph, Node, Edge
from algorithms import PrimMST
from algorithms.bridges import find_bridges, is_2_edge_connected, count_bridges
from algorithms.augmentation import augment_to_2_edge_connected
from data.generator import GraphGenerator
from data.demo import create_demo_graph


class TestTarjanBridges:

    def test_chain_all_edges_are_bridges(self):
        """Ланцюг 0-1-2-3-4: усі 4 ребра — мости."""
        g = Graph()
        for i in range(5):
            g.add_node(Node(i, f"N{i}", 48.0 + i * 0.01, 35.0))
        for i in range(4):
            g.add_edge(Edge(i, i + 1, 1000.0, 1.0))
        bridges = find_bridges(g)
        assert len(bridges) == 4
        assert not is_2_edge_connected(g)

    def test_cycle_no_bridges(self):
        """Цикл 0-1-2-3-0: жодного моста."""
        g = Graph()
        for i in range(4):
            g.add_node(Node(i, f"N{i}", 48.0 + i * 0.01, 35.0))
        edges = [(0, 1), (1, 2), (2, 3), (0, 3)]
        for a, b in edges:
            g.add_edge(Edge(a, b, 1000.0, 1.0))
        assert find_bridges(g) == []
        assert is_2_edge_connected(g)

    def test_two_triangles_bridge(self):
        """Два трикутники з'єднані одним ребром — це ребро є мостом."""
        g = Graph()
        for i in range(6):
            g.add_node(Node(i, f"N{i}", 48.0 + i * 0.01, 35.0))
        # Лівий трикутник
        for a, b in [(0, 1), (1, 2), (0, 2)]:
            g.add_edge(Edge(a, b, 1.0, 1.0))
        # Правий трикутник
        for a, b in [(3, 4), (4, 5), (3, 5)]:
            g.add_edge(Edge(a, b, 1.0, 1.0))
        # Міст
        g.add_edge(Edge(2, 3, 1.0, 1.0))

        bridges = find_bridges(g)
        assert len(bridges) == 1
        b = bridges[0]
        assert {b.node1, b.node2} == {2, 3}

    def test_empty_graph(self):
        assert find_bridges(Graph()) == []
        assert is_2_edge_connected(Graph())

    def test_single_node(self):
        g = Graph()
        g.add_node(Node(0, "A", 48.0, 35.0))
        assert find_bridges(g) == []
        assert is_2_edge_connected(g)

    def test_two_nodes_one_edge_is_bridge(self):
        g = Graph()
        g.add_node(Node(0, "A", 48.0, 35.0))
        g.add_node(Node(1, "B", 48.1, 35.1))
        g.add_edge(Edge(0, 1, 1000.0, 1.0))
        bridges = find_bridges(g)
        assert len(bridges) == 1

    def test_star_topology_all_bridges(self):
        """Зірка: центр з'єднаний з 5 листками. Усі 5 ребер — мости."""
        g = Graph()
        g.add_node(Node(0, "Center", 48.5, 35.5))
        for i in range(1, 6):
            g.add_node(Node(i, f"Leaf{i}", 48.5 + i * 0.01, 35.5))
            g.add_edge(Edge(0, i, 1000.0, 1.0))
        assert len(find_bridges(g)) == 5

    def test_mst_every_edge_is_bridge(self):
        """У МКД (яке є деревом) кожне ребро є мостом."""
        g = create_demo_graph()
        mst = PrimMST(g).find_mst()
        # Граф тільки з МКД-ребер
        mst_graph = Graph()
        for n in g.iter_nodes():
            mst_graph.add_node(Node(n.id, n.name, n.x, n.y))
        for e in mst.edges:
            mst_graph.add_edge(Edge(e.node1, e.node2, e.distance, e.cost_per_meter))

        bridges = find_bridges(mst_graph)
        assert len(bridges) == mst_graph.edge_count()
        assert len(bridges) == g.node_count() - 1
        assert not is_2_edge_connected(mst_graph)

    def test_complete_graph_no_bridges(self):
        """K_n при n>=3 — 2-EC."""
        g = Graph()
        for i in range(5):
            g.add_node(Node(i, f"N{i}", 48.0 + i * 0.01, 35.0))
        g.generate_complete_graph(cost_per_meter=1.0)
        assert find_bridges(g) == []
        assert is_2_edge_connected(g)

    def test_count_bridges_matches_find(self):
        g = create_demo_graph()
        mst = PrimMST(g).find_mst()
        mst_graph = Graph()
        for n in g.iter_nodes():
            mst_graph.add_node(Node(n.id, n.name, n.x, n.y))
        for e in mst.edges:
            mst_graph.add_edge(Edge(e.node1, e.node2, e.distance, e.cost_per_meter))
        assert count_bridges(mst_graph) == len(find_bridges(mst_graph))


class TestAugmentation:

    def _check_invariant(self, graph: Graph, mst_edges, result):
        """МКД + reserve_edges має бути 2-EC і вкладатись у 2-апроксимацію."""
        combined = Graph()
        for n in graph.iter_nodes():
            combined.add_node(Node(n.id, n.name, n.x, n.y))
        for e in mst_edges:
            combined.add_edge(Edge(e.node1, e.node2, e.distance, e.cost_per_meter))
        for e in result.reserve_edges:
            if not combined.has_edge(e.node1, e.node2):
                combined.add_edge(Edge(e.node1, e.node2, e.distance, e.cost_per_meter))

        if graph.node_count() >= 3:
            assert is_2_edge_connected(combined), (
                f"Після доповнення граф НЕ 2-EC: "
                f"{count_bridges(combined)} мостів залишилось"
            )
        assert result.is_2_edge_connected
        assert result.bridges_remaining == 0

        # нижня межа ⌈L/2⌉ де L — листки
        leaves = sum(1 for nid in graph.get_node_ids()
                     if sum(1 for e in mst_edges
                            if e.node1 == nid or e.node2 == nid) == 1)
        from math import ceil
        assert result.lower_bound == ceil(leaves / 2)

        if result.lower_bound > 0:
            assert result.edges_added <= 2 * result.lower_bound, (
                f"Гірше за 2-апроксимацію: "
                f"{result.edges_added} > 2*{result.lower_bound}"
            )

    def test_demo_25_ps(self):
        g = create_demo_graph()
        mst = PrimMST(g).find_mst()
        result = augment_to_2_edge_connected(g, mst.edges)
        assert result.edges_added >= result.lower_bound
        self._check_invariant(g, mst.edges, result)

    @pytest.mark.parametrize("seed,V", [(42, 30), (7, 50), (99, 100)])
    def test_random_complete(self, seed, V):
        g = GraphGenerator(seed=seed).random_graph(V, complete=True)
        mst = PrimMST(g, record_steps=False).find_mst()
        result = augment_to_2_edge_connected(g, mst.edges)
        self._check_invariant(g, mst.edges, result)

    @pytest.mark.parametrize("seed,V,E", [(42, 40, 120), (7, 60, 200)])
    def test_sparse_graph(self, seed, V, E):
        """Розріджений граф (як у рецензента)."""
        g = GraphGenerator(seed=seed).sparse_graph(num_nodes=V, num_edges=E)
        mst = PrimMST(g, record_steps=False).find_mst()
        result = augment_to_2_edge_connected(g, mst.edges)
        self._check_invariant(g, mst.edges, result)

    @pytest.mark.parametrize("rows,cols", [(4, 4), (3, 5)])
    def test_grid_graph(self, rows, cols):
        g = GraphGenerator(seed=42).grid_graph(rows=rows, cols=cols)
        mst = PrimMST(g, record_steps=False).find_mst()
        result = augment_to_2_edge_connected(g, mst.edges)
        self._check_invariant(g, mst.edges, result)

    def test_two_nodes_no_augmentation_possible(self):
        """V=2: 2-EC недосяжна без кратних ребер. Має не падати."""
        g = Graph()
        g.add_node(Node(0, "A", 48.0, 35.0))
        g.add_node(Node(1, "B", 48.1, 35.1))
        g.add_edge(Edge(0, 1, 1000.0, 1.0))
        mst = PrimMST(g).find_mst()
        result = augment_to_2_edge_connected(g, mst.edges)
        assert result.leaves_count == 2
        assert result.lower_bound == 1

    def test_result_contains_metrics(self):
        """AugmentationResult має заповнюватись усіма полями."""
        g = create_demo_graph()
        mst = PrimMST(g).find_mst()
        result = augment_to_2_edge_connected(g, mst.edges, track_memory=True)

        assert result.execution_time > 0
        assert result.peak_memory_bytes is not None and result.peak_memory_bytes > 0
        assert result.peak_memory_kb > 0
        assert result.leaves_count > 0
        assert result.lower_bound > 0
        assert result.reserve_cost > 0
        assert result.reserve_distance > 0
        assert result.approximation_ratio >= 1.0

    def test_track_memory_off(self):
        """Без tracemalloc peak_memory_bytes = None."""
        g = create_demo_graph()
        mst = PrimMST(g).find_mst()
        result = augment_to_2_edge_connected(g, mst.edges, track_memory=False)
        assert result.peak_memory_bytes is None
        assert result.execution_time > 0
