"""
Пошук мостів у графі (алгоритм Тар'яна).

Міст — ребро, без якого граф розпадається на компоненти. У дереві
(зокрема МКД) кожне ребро — міст.

DFS ітеративна, бо для великих V  рекурсія може упертись у дефолтний
ліміт стека Python на ланцюгових графах.
"""

from typing import List, Dict, Set, Tuple, Optional

from models import Graph, Edge


def find_bridges(graph: Graph) -> List[Edge]:
    """Усі мости графа. Для зв'язного 2-EC графа — порожній список."""
    if graph.node_count() == 0:
        return []

    disc: Dict[int, int] = {}
    low: Dict[int, int] = {}
    bridges: List[Edge] = []
    timer = [0]  # list для мутабельності з внутрішньої функції

    for start in graph.get_node_ids():
        if start not in disc:
            _dfs_iterative(graph, start, disc, low, bridges, timer)

    return bridges


def _dfs_iterative(graph: Graph,
                   start: int,
                   disc: Dict[int, int],
                   low: Dict[int, int],
                   bridges: List[Edge],
                   timer: List[int]) -> None:
    """Кадр стека: (vertex, parent_edge, neighbor_iter). parent_edge — щоб
    не плутати з back-edge на кратних ребрах."""
    disc[start] = low[start] = timer[0]
    timer[0] += 1

    # Стек містить кадри: (vertex, parent_edge, iter_of_neighbor_edges)
    stack: List[Tuple[int, Optional[Edge], iter]] = [
        (start, None, iter(graph.get_neighbors(start)))
    ]

    while stack:
        v, parent_edge, edges_iter = stack[-1]
        next_edge = next(edges_iter, None)

        if next_edge is None:
            # сусідів більше нема — повертаємось до батька і оновлюємо його low
            stack.pop()
            if stack:
                parent_v = stack[-1][0]
                if low[v] < low[parent_v]:
                    low[parent_v] = low[v]
                if parent_edge is not None and low[v] > disc[parent_v]:
                    bridges.append(parent_edge)
            continue

        u = next_edge.get_other_node(v)

        # не повертаємось тим самим ребром (для кратних ребер)
        if parent_edge is not None and next_edge is parent_edge:
            continue

        if u not in disc:
            disc[u] = low[u] = timer[0]
            timer[0] += 1
            stack.append((u, next_edge, iter(graph.get_neighbors(u))))
        else:
            # back-edge — підтягуємо low
            if disc[u] < low[v]:
                low[v] = disc[u]


def is_2_edge_connected(graph: Graph) -> bool:
    """True, якщо граф зв'язний і не має мостів."""
    if graph.node_count() <= 1:
        return True
    if not graph.is_connected():
        return False
    return len(find_bridges(graph)) == 0


def count_bridges(graph: Graph) -> int:
    return len(find_bridges(graph))
