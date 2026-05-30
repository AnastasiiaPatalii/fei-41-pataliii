"""
Доповнення МКД до 2-edge-connected графа.

Tree Augmentation Problem (TAP) — у загальному випадку NP-важка.
Тут — 2-апроксимація: парування листків + жадібне покриття залишкових
мостів. Нижня межа додаваних ребер — ⌈L/2⌉, де L — кількість листків.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional
from collections import deque
import gc
import math
import time
import tracemalloc

from models import Graph, Node, Edge
from .bridges import find_bridges, is_2_edge_connected


@dataclass
class AugmentationResult:
    reserve_edges: List[Edge] = field(default_factory=list)
    reserve_cost: float = 0.0
    reserve_distance: float = 0.0
    execution_time: float = 0.0
    peak_memory_bytes: Optional[int] = None
    leaves_count: int = 0
    lower_bound: int = 0  # ⌈L/2⌉
    bridges_remaining: int = 0
    is_2_edge_connected: bool = False

    @property
    def edges_added(self) -> int:
        return len(self.reserve_edges)

    @property
    def peak_memory_kb(self) -> float:
        return (self.peak_memory_bytes or 0) / 1024

    @property
    def approximation_ratio(self) -> float:
        """1.0 — оптимум; 2.0 — удвічі гірше."""
        if self.lower_bound == 0:
            return 1.0
        return self.edges_added / self.lower_bound

    def __str__(self) -> str:
        return (
            f"Доповнення до 2-edge-connected:\n"
            f"  Додано ребер: {self.edges_added} (нижня межа: {self.lower_bound})\n"
            f"  Коефіцієнт апроксимації: {self.approximation_ratio:.2f}x від оптимуму\n"
            f"  Сумарна довжина резерву: {self.reserve_distance:,.0f} м\n"
            f"  Сумарна вартість резерву: {self.reserve_cost:,.2f} грн\n"
            f"  Час: {self.execution_time*1000:.2f} мс\n"
            f"  Пам'ять (пік): {self.peak_memory_kb:,.1f} КБ\n"
            f"  Залишилось мостів: {self.bridges_remaining}\n"
            f"  Граф 2-edge-connected: {self.is_2_edge_connected}"
        )


def augment_to_2_edge_connected(graph: Graph,
                                mst_edges: List[Edge],
                                track_memory: bool = True) -> AugmentationResult:
    """Додає резервні ребра до МКД щоб у графі не залишилось мостів.

    Якщо у graph немає потрібного ребра — створюємо синтетичне
    (відстань — гаверсинус, cost_per_meter — середнє по МКД).
    """
    peak_mem = None
    if track_memory:
        was_tracing = tracemalloc.is_tracing()
        if not was_tracing:
            tracemalloc.start()
        else:
            tracemalloc.clear_traces()

    start_time = time.perf_counter()

    # тільки ребра МКД — для DFS по дереву і пошуку мостів
    tree_graph = Graph()
    for node in graph.iter_nodes():
        tree_graph.add_node(Node(node.id, node.name, node.x, node.y))
    for e in mst_edges:
        tree_graph.add_edge(Edge(e.node1, e.node2, e.distance, e.cost_per_meter))

    leaves = [nid for nid in tree_graph.get_node_ids()
              if tree_graph.get_degree(nid) == 1]
    leaves_count = len(leaves)
    lower_bound = math.ceil(leaves_count / 2) if leaves_count > 0 else 0

    mst_keys: Set[Tuple[int, int]] = set()
    for e in mst_edges:
        mst_keys.add((min(e.node1, e.node2), max(e.node1, e.node2)))

    # неMST-ребра з вихідного графа — пул кандидатів
    candidate_pool: Dict[Tuple[int, int], Edge] = {}
    for e in graph.iter_edges():
        key = (min(e.node1, e.node2), max(e.node1, e.node2))
        if key not in mst_keys:
            candidate_pool[key] = e

    avg_cpm = (sum(e.cost_per_meter for e in mst_edges) / len(mst_edges)
               if mst_edges else 150.0)

    reserve_edges: List[Edge] = []

    # --- Фаза 1: парування листків ---
    # V<3: 2-EC у простому графі недосяжна (треба multigraph) — пропускаємо
    if leaves_count >= 2 and graph.node_count() >= 3:
        leaves_dfs_order = _dfs_leaf_order(tree_graph, leaves)
        k = len(leaves_dfs_order)
        half = k // 2
        # Frederickson-JáJá: i-й листок з (i+half)-м
        for i in range(half):
            u = leaves_dfs_order[i]
            v = leaves_dfs_order[i + half]
            pair_key = (min(u, v), max(u, v))
            if pair_key in mst_keys:
                continue
            edge = _find_or_create_edge(graph, u, v, candidate_pool,
                                        mst_keys, avg_cpm)
            reserve_edges.append(edge)
            mst_keys.add(pair_key)
        if k % 2 == 1:
            u = leaves_dfs_order[-1]
            v = leaves_dfs_order[0]
            pair_key = (min(u, v), max(u, v))
            if pair_key not in mst_keys:
                edge = _find_or_create_edge(graph, u, v, candidate_pool,
                                            mst_keys, avg_cpm)
                reserve_edges.append(edge)
                mst_keys.add(pair_key)

    # --- Фаза 2: жадібне покриття залишкових мостів ---
    current_graph = Graph()
    for node in graph.iter_nodes():
        current_graph.add_node(Node(node.id, node.name, node.x, node.y))
    for e in mst_edges:
        current_graph.add_edge(Edge(e.node1, e.node2, e.distance, e.cost_per_meter))
    for e in reserve_edges:
        current_graph.add_edge(Edge(e.node1, e.node2, e.distance, e.cost_per_meter))

    safety_iter = 0
    max_iter = max(50, graph.node_count())
    while safety_iter < max_iter and graph.node_count() >= 3:
        safety_iter += 1
        bridges = find_bridges(current_graph)
        if not bridges:
            break

        bridge_set = {(min(b.node1, b.node2), max(b.node1, b.node2))
                      for b in bridges}

        best_edge: Optional[Edge] = None
        best_cost = float('inf')

        # перший знайдений найдешевший, що покриває хоч один міст
        sorted_cands = sorted(candidate_pool.values(), key=lambda e: e.weight)
        for cand in sorted_cands:
            cand_key = (min(cand.node1, cand.node2),
                        max(cand.node1, cand.node2))
            if cand_key in mst_keys:
                continue
            path_edges = _tree_path_edges(current_graph, cand.node1, cand.node2)
            covers_any = any(
                (min(pe.node1, pe.node2), max(pe.node1, pe.node2)) in bridge_set
                for pe in path_edges
            )
            if covers_any:
                if cand.weight < best_cost:
                    best_edge = cand
                    best_cost = cand.weight
                    break  # sorted — перший знайдений і є найдешевший

        if best_edge is None:
            # Кандидатів немає. На розрідженому графі генеруємо синтетичне
            # ребро на "інший" кінець моста.
            b = bridges[0]
            other = _farthest_via_bridge(current_graph, b)
            if other == b.node1 or other == b.node2:
                break  # вироджений випадок (наприклад чистий ланцюг V=2..3)
            new_key = (min(b.node1, other), max(b.node1, other))
            if new_key in mst_keys:
                break  # дублікат — мульти-ребра ми не підтримуємо
            new_edge = _find_or_create_edge(graph, b.node1, other,
                                            candidate_pool, mst_keys, avg_cpm)
            reserve_edges.append(new_edge)
            current_graph.add_edge(Edge(new_edge.node1, new_edge.node2,
                                        new_edge.distance, new_edge.cost_per_meter))
            mst_keys.add(new_key)
            continue

        reserve_edges.append(best_edge)
        current_graph.add_edge(Edge(best_edge.node1, best_edge.node2,
                                    best_edge.distance, best_edge.cost_per_meter))
        mst_keys.add((min(best_edge.node1, best_edge.node2),
                      max(best_edge.node1, best_edge.node2)))

    final_bridges = find_bridges(current_graph)
    execution_time = time.perf_counter() - start_time

    if track_memory:
        _, peak_mem = tracemalloc.get_traced_memory()
        if not was_tracing:
            tracemalloc.stop()

    return AugmentationResult(
        reserve_edges=reserve_edges,
        reserve_cost=sum(e.weight for e in reserve_edges),
        reserve_distance=sum(e.distance for e in reserve_edges),
        execution_time=execution_time,
        peak_memory_bytes=peak_mem,
        leaves_count=leaves_count,
        lower_bound=lower_bound,
        bridges_remaining=len(final_bridges),
        is_2_edge_connected=(len(final_bridges) == 0 and current_graph.is_connected()),
    )


def _dfs_leaf_order(tree: Graph, leaves: List[int]) -> List[int]:
    """Листки у порядку DFS — щоб "протилежні" листки парувались разом."""
    if not leaves:
        return []
    leaf_set = set(leaves)
    # стартуємо з не-листка щоб не застрягти у першому ж піддереві
    start = next((nid for nid in tree.get_node_ids()
                  if nid not in leaf_set), leaves[0])

    ordered: List[int] = []
    visited: Set[int] = set()
    stack = [start]
    while stack:
        v = stack.pop()
        if v in visited:
            continue
        visited.add(v)
        if v in leaf_set:
            ordered.append(v)
        # Додаємо сусідів у стек (порядок не суворо детермінований
        # для прискорення; для нашої задачі цього достатньо)
        for edge in tree.get_neighbors(v):
            u = edge.get_other_node(v)
            if u not in visited:
                stack.append(u)
    return ordered


def _tree_path_edges(tree: Graph, u: int, v: int) -> List[Edge]:
    """Ребра на шляху u → v у дереві (шлях єдиний)."""
    if u == v:
        return []
    parent: Dict[int, Tuple[int, Edge]] = {}
    queue = deque([u])
    visited = {u}
    while queue:
        cur = queue.popleft()
        if cur == v:
            break
        for edge in tree.get_neighbors(cur):
            nb = edge.get_other_node(cur)
            if nb not in visited:
                visited.add(nb)
                parent[nb] = (cur, edge)
                queue.append(nb)
    if v not in parent and v != u:
        return []  # незв'язно
    path: List[Edge] = []
    cur = v
    while cur in parent:
        prev, edge = parent[cur]
        path.append(edge)
        cur = prev
    return path


def _farthest_via_bridge(graph: Graph, bridge: Edge) -> int:
    """Найвіддаленіша вершина з протилежного боку моста. Запасний варіант
    для Фази 2 коли кандидатів немає."""
    # BFS з node1 без переходу через сам bridge
    visited = {bridge.node1}
    queue = deque([bridge.node1])
    farthest = bridge.node1
    while queue:
        cur = queue.popleft()
        for edge in graph.get_neighbors(cur):
            if edge is bridge:
                continue
            nb = edge.get_other_node(cur)
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)
                farthest = nb
    # якщо node2 не дістали — він на іншому боці, BFS звідти
    if bridge.node2 not in visited:
        visited2 = {bridge.node2}
        queue = deque([bridge.node2])
        farthest = bridge.node2
        while queue:
            cur = queue.popleft()
            for edge in graph.get_neighbors(cur):
                if edge is bridge:
                    continue
                nb = edge.get_other_node(cur)
                if nb not in visited2:
                    visited2.add(nb)
                    queue.append(nb)
                    farthest = nb
    return farthest


def _find_or_create_edge(graph: Graph, u: int, v: int,
                         candidate_pool: Dict[Tuple[int, int], Edge],
                         mst_keys: Set[Tuple[int, int]],
                         avg_cpm: float) -> Edge:
    """Бере ребро (u, v) з пулу або синтезує нове з гаверсинусом."""
    if u == v:
        raise ValueError(f"Не можна створити петлю на вершині {u}")
    key = (min(u, v), max(u, v))
    if key in candidate_pool:
        return candidate_pool[key]
    # Синтезуємо ребро з координат вершин
    n1 = graph.get_node(u)
    n2 = graph.get_node(v)
    dist = n1.distance_to(n2)
    return Edge(u, v, dist, avg_cpm)
