"""
Перевіряє, що Binary / d-ary / Indexed варіанти Пріма дають МКД з
однаковою сумарною вагою. Самі ребра можуть відрізнятись (теорема про
єдиність МКД виконується лише при різних вагах), але cost має збігатись.
"""

import pytest

from models import Graph, Node, Edge
from algorithms import PrimMST, PrimMSTDAry, PrimMSTIndexed, KruskalMST
from data.generator import GraphGenerator


PRIM_VARIANTS = [
    ("binary", lambda g: PrimMST(g, record_steps=False)),
    ("2-ary", lambda g: PrimMSTDAry(g, record_steps=False, d=2)),
    ("3-ary", lambda g: PrimMSTDAry(g, record_steps=False, d=3)),
    ("4-ary", lambda g: PrimMSTDAry(g, record_steps=False, d=4)),
    ("8-ary", lambda g: PrimMSTDAry(g, record_steps=False, d=8)),
    ("indexed", lambda g: PrimMSTIndexed(g, record_steps=False)),
]


def _mst_costs_match(graph: Graph) -> None:
    """Всі варіанти Пріма + Крускал дають однакову сумарну вагу."""
    expected_edges = graph.node_count() - 1 if graph.is_connected() else None
    reference = KruskalMST(graph, record_steps=False).find_mst()

    for name, factory in PRIM_VARIANTS:
        result = factory(graph).find_mst()
        if expected_edges is not None:
            assert len(result.edges) == expected_edges, (
                f"{name}: ребер={len(result.edges)}, очікувалось {expected_edges}"
            )
        covered = set()
        for e in result.edges:
            covered.add(e.node1)
            covered.add(e.node2)
        if graph.is_connected():
            assert covered == set(graph.get_node_ids()), (
                f"{name}: МКД не покриває всі вершини"
            )
        # Збіг сумарної вартості (з Крускалом як референсом)
        assert abs(result.total_cost - reference.total_cost) < 1e-6, (
            f"{name}: cost={result.total_cost}, kruskal={reference.total_cost}"
        )


# ──────────────────────────────────────────────────────────────────────
# Топології
# ──────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("seed,size", [(42, 30), (7, 50), (99, 100)])
def test_random_complete_graph(seed, size):
    """Випадковий повний граф."""
    gen = GraphGenerator(seed=seed)
    g = gen.random_graph(size, complete=True)
    _mst_costs_match(g)


@pytest.mark.parametrize("seed,nodes,edges", [(42, 50, 200), (7, 100, 400)])
def test_sparse_graph(seed, nodes, edges):
    gen = GraphGenerator(seed=seed)
    g = gen.sparse_graph(num_nodes=nodes, num_edges=edges)
    _mst_costs_match(g)


@pytest.mark.parametrize("rows,cols", [(4, 5), (6, 6), (3, 10)])
def test_grid_graph(rows, cols):
    gen = GraphGenerator(seed=42)
    g = gen.grid_graph(rows=rows, cols=cols)
    _mst_costs_match(g)


@pytest.mark.parametrize("clusters,per", [(3, 6), (5, 8)])
def test_cluster_graph(clusters, per):
    gen = GraphGenerator(seed=42)
    g = gen.cluster_graph(num_clusters=clusters, nodes_per_cluster=per)
    _mst_costs_match(g)


# --- граничні випадки ---

def test_two_nodes():
    """V=2 — мінімальний нетривіальний граф."""
    g = Graph()
    g.add_node(Node(0, "A", 48.0, 35.0))
    g.add_node(Node(1, "B", 48.5, 35.5))
    g.add_edge(Edge(0, 1, 1000.0, 150.0))
    _mst_costs_match(g)


def test_single_node():
    """Граф з однієї вершини — МКД порожнє."""
    g = Graph()
    g.add_node(Node(0, "A", 48.0, 35.0))
    for name, factory in PRIM_VARIANTS:
        r = factory(g).find_mst()
        assert r.edges == [], f"{name}: для V=1 МКД має бути порожнім"


def test_equal_weights():
    """Рівні ваги — конкретні ребра можуть відрізнятись, але cost однаковий."""
    g = Graph()
    for i in range(5):
        g.add_node(Node(i, f"N{i}", 48.0 + i * 0.001, 35.0))
    edges = [(0, 1), (0, 2), (1, 2), (2, 3), (3, 4), (1, 4)]
    for a, b in edges:
        g.add_edge(Edge(a, b, 1000.0, 150.0))
    _mst_costs_match(g)


# --- юніт-тести для структур даних ---

def test_indexed_pq_operations():
    from algorithms.prim_indexed import IndexedMinPQ

    pq = IndexedMinPQ()
    pq.insert(1, 5.0)
    pq.insert(2, 3.0)
    pq.insert(3, 7.0)

    assert len(pq) == 3
    assert pq.contains(2)
    assert not pq.contains(99)

    # після decrease_key елемент 3 має стати мінімумом
    pq.decrease_key(3, 1.0)
    assert pq.extract_min() == 3
    assert pq.extract_min() == 2
    assert pq.extract_min() == 1
    assert len(pq) == 0


def test_indexed_pq_decrease_key_invalid():
    from algorithms.prim_indexed import IndexedMinPQ

    pq = IndexedMinPQ()
    pq.insert(1, 5.0)
    with pytest.raises(ValueError):
        pq.decrease_key(1, 10.0)


def test_dary_heap_basic():
    """pop має повертати елементи за зростанням."""
    from algorithms.prim_dary import DAryHeap
    from models import Edge

    edges = [Edge(0, 1, 100, 1.0), Edge(0, 2, 50, 1.0), Edge(0, 3, 200, 1.0)]

    for d in [2, 3, 4, 8]:
        heap = DAryHeap(d=d)
        for e in edges:
            heap.push((e.weight, e))

        popped = []
        while heap:
            _, e = heap.pop()
            popped.append(e.weight)
        assert popped == sorted(popped), f"d={d}: не відсортовано — {popped}"
