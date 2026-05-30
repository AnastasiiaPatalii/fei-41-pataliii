"""
test_cross_validation.py — Крос-валідація: Пріма та Крускала

Рівень тестування: Regression (регресійне)
Мета: Довести, що два різних алгоритми (Пріма з бінарною купою та
      Крускал з Union-Find) завжди знаходять МКД однакової вартості
      на одному й тому ж графі, незалежно від розміру, топології та
      розподілу ваг.

Стратегія тестування:
    - Параметризація за розміром (5, 15, 50 вершин)
    - Параметризація за seed (42, 123, 999) — різні конфігурації
    - Різні топології: повний граф, кластерний, сітковий
    - Різні вартості: від дешевих до дорогих ліній

pytest.approx(rel=1e-5) — допуск на похибку float.
"""

import pytest
from data.generator import GraphGenerator
from algorithms import PrimMST, KruskalMST


# ПОВНІ ВИПАДКОВІ ГРАФИ

@pytest.mark.parametrize("num_nodes", [3, 5, 15, 50])
@pytest.mark.parametrize("seed", [42, 123, 999])
def test_prim_vs_kruskal_random_complete(num_nodes, seed):
    """
    Крос-валідація на випадкових повних графах.
    Повний граф K_n — найщільніша топологія, максимальний вибір ребер.
    """
    gen = GraphGenerator(seed=seed)
    graph = gen.random_graph(
        num_nodes=num_nodes,
        cost_per_meter=150.0,
        complete=True
    )

    prim_result = PrimMST(graph, record_steps=False).find_mst()
    kruskal_result = KruskalMST(graph, record_steps=False).find_mst()

    # Вартість має збігатися
    assert prim_result.total_cost == pytest.approx(
        kruskal_result.total_cost, rel=1e-5
    ), (
        f"V={num_nodes}, seed={seed}: "
        f"Пріма={prim_result.total_cost:.2f}, "
        f"Крускал={kruskal_result.total_cost:.2f}"
    )

    # Кількість ребер має збігатися
    assert len(prim_result.edges) == len(kruskal_result.edges)


# КЛАСТЕРНІ ГРАФИ

@pytest.mark.parametrize("seed", [10, 20, 30])
def test_prim_vs_kruskal_cluster(seed):
    """
    Крос-валідація на кластерних графах.
    Кластерна топологія імітує реальні енергомережі:
    підстанції згруповані навколо населених пунктів.
    """
    gen = GraphGenerator(seed=seed)
    graph = gen.cluster_graph(
        num_clusters=3,
        nodes_per_cluster=6,
        cost_per_meter=150.0
    )

    prim_cost = PrimMST(graph, record_steps=False).find_mst().total_cost
    kruskal_cost = KruskalMST(graph, record_steps=False).find_mst().total_cost

    assert prim_cost == pytest.approx(kruskal_cost, rel=1e-5)


# СІТКОВІ ГРАФИ (розріджені)

@pytest.mark.parametrize("rows,cols", [(3, 3), (4, 4), (5, 5)])
def test_prim_vs_kruskal_grid(rows, cols):
    """
    Крос-валідація на графі-сітці.
    Сітка — розріджений граф (degree ≤ 4), тестує алгоритми
    в умовах, коли вибір ребер обмежений.
    """
    gen = GraphGenerator(seed=1)
    graph = gen.grid_graph(rows=rows, cols=cols, cost_per_meter=150.0)

    prim_cost = PrimMST(graph, record_steps=False).find_mst().total_cost
    kruskal_cost = KruskalMST(graph, record_steps=False).find_mst().total_cost

    assert prim_cost == pytest.approx(kruskal_cost, rel=1e-5)

# РІЗНІ ВАРТОСТІ ЗА МЕТР

@pytest.mark.parametrize("cost_per_meter", [50.0, 150.0, 500.0, 1200.0])
def test_prim_vs_kruskal_different_costs(cost_per_meter):
    """
    Зміна cost_per_meter масштабує всі ваги рівномірно,
    тому порядок ребер не змінюється і МКД залишається тим самим.
    Перевіряємо, що обидва алгоритми коректно масштабують.
    """
    gen = GraphGenerator(seed=42)
    graph = gen.random_graph(
        num_nodes=12,
        cost_per_meter=cost_per_meter,
        complete=True
    )

    prim_cost = PrimMST(graph, record_steps=False).find_mst().total_cost
    kruskal_cost = KruskalMST(graph, record_steps=False).find_mst().total_cost

    assert prim_cost == pytest.approx(kruskal_cost, rel=1e-5)


# ТАКОЖ ПЕРЕВІРЯЄМО ДОВЖИНУ (total_distance)

@pytest.mark.parametrize("num_nodes", [8, 20])
def test_prim_vs_kruskal_distance_match(num_nodes):
    """
    Окрім вартості, загальна довжина МКД також має збігатися.
    Якщо вартість однакова, але довжини різні — це помилка
    в обчисленні total_distance.
    """
    gen = GraphGenerator(seed=55)
    graph = gen.random_graph(
        num_nodes=num_nodes,
        cost_per_meter=150.0,
        complete=True
    )

    prim = PrimMST(graph, record_steps=False).find_mst()
    kruskal = KruskalMST(graph, record_steps=False).find_mst()

    assert prim.total_distance == pytest.approx(kruskal.total_distance, rel=1e-5)


# СТАБІЛЬНІСТЬ ПРИ ПОВТОРНИХ ЗАПУСКАХ

def test_prim_deterministic():
    """
    Повторний запуск Пріма на тому ж графі дає той самий результат.
    Важливо для відтворюваності результатів у курсовій.
    """
    gen = GraphGenerator(seed=42)
    graph = gen.random_graph(num_nodes=20, cost_per_meter=150.0, complete=True)

    cost1 = PrimMST(graph, record_steps=False).find_mst().total_cost
    cost2 = PrimMST(graph, record_steps=False).find_mst().total_cost

    assert cost1 == pytest.approx(cost2)


def test_kruskal_deterministic():
    """Повторний запуск Крускала на тому ж графі дає той самий результат."""
    gen = GraphGenerator(seed=42)
    graph = gen.random_graph(num_nodes=20, cost_per_meter=150.0, complete=True)

    cost1 = KruskalMST(graph, record_steps=False).find_mst().total_cost
    cost2 = KruskalMST(graph, record_steps=False).find_mst().total_cost

    assert cost1 == pytest.approx(cost2)