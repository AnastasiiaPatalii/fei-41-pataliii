"""
Динамічне оновлення МКД при змінах топології (додавання/видалення
вершини, зміна ваги ребра).

Додавання — інкрементально за O(V). Видалення і зміна ваги — через
повну перебудову Прімом за O((V+E) log V).
"""

from typing import List, Optional, Set, Tuple, Dict
from dataclasses import dataclass, field
import time

from .base_mst import BaseMST, MSTResult
from .prim import PrimMST
from models import Graph, Node, Edge


@dataclass
class UpdateResult:
    """
    Результат операції оновлення МКД.

    operation — 'add_node' / 'remove_node' / 'update_edge'.
    rebuild_required — чи довелось повністю перебудовувати МКД.
    """
    operation: str
    success: bool
    mst_edges: List[Edge]
    total_cost: float
    total_distance: float
    execution_time: float
    rebuild_required: bool
    affected_edges: List[Edge] = field(default_factory=list)
    message: str = ""

    def __str__(self) -> str:
        status = "✓" if self.success else "✗"
        rebuild = " (перебудова)" if self.rebuild_required else " (інкрементально)"
        return (
            f"{status} {self.operation}{rebuild}\n"
            f"   Ребер у МКД: {len(self.mst_edges)}\n"
            f"   Вартість: {self.total_cost:,.2f} грн\n"
            f"   Час: {self.execution_time * 1000:.3f} мс"
        )


class DynamicMST:
    """Тримає МКД актуальним при змінах графа.

    add_node — інкрементально за O(E_new).
    remove_node, update_edge_cost — повна перебудова Прімом.
    """

    def __init__(self, graph: Graph):
        self.graph = graph
        self._mst_edges: List[Edge] = []
        self._mst_edge_set: Set[Tuple[int, int]] = set()
        self._history: List[UpdateResult] = []

        # Будуємо початкове МКД
        self._rebuild()

    @property
    def mst_edges(self) -> List[Edge]:
        """Повертає копію поточних ребер МКД."""
        return list(self._mst_edges)

    @property
    def total_cost(self) -> float:
        """Повертає загальну вартість МКД."""
        return sum(e.weight for e in self._mst_edges)

    @property
    def total_distance(self) -> float:
        """Повертає загальну довжину МКД."""
        return sum(e.distance for e in self._mst_edges)

    @property
    def history(self) -> List[UpdateResult]:
        """Повертає історію операцій."""
        return list(self._history)

    def _rebuild(self) -> float:
        """Повна перебудова МКД. Повертає час."""
        start = time.perf_counter()

        if self.graph.node_count() < 2:
            self._mst_edges = []
            self._mst_edge_set = set()
        else:
            prim = PrimMST(self.graph, record_steps=False)
            result = prim.find_mst()
            self._mst_edges = result.edges
            self._mst_edge_set = {(e.node1, e.node2) for e in self._mst_edges}

        return time.perf_counter() - start

    def _is_in_mst(self, node1: int, node2: int) -> bool:
        """Перевіряє, чи ребро в поточному МКД."""
        key = (min(node1, node2), max(node1, node2))
        return key in self._mst_edge_set

    def add_node(self,
                 node: Node,
                 cost_per_meter: float = 100.0,
                 connect_to: Optional[List[int]] = None) -> UpdateResult:
        """Додає вершину інкрементально: додаємо ребра до connect_to і
        приєднуємо нову вершину до МКД через найдешевше з них."""
        start = time.perf_counter()

        if self.graph.has_node(node.id):
            return UpdateResult(
                operation="add_node",
                success=False,
                mst_edges=self.mst_edges,
                total_cost=self.total_cost,
                total_distance=self.total_distance,
                execution_time=time.perf_counter() - start,
                rebuild_required=False,
                message=f"Вершина {node.id} вже існує"
            )

        # Визначаємо до яких вершин підключати
        if connect_to is None:
            connect_to = self.graph.get_node_ids()

        # Додаємо вершину
        self.graph.add_node(node)

        # ребра з нової вершини до connect_to
        new_edges = []
        for other_id in connect_to:
            if other_id == node.id:
                continue

            other_node = self.graph.get_node(other_id)
            distance = node.distance_to(other_node)
            edge = Edge(node.id, other_id, distance, cost_per_meter)
            self.graph.add_edge(edge)
            new_edges.append(edge)

        if not new_edges:
            # ізольована вершина — нема чим приєднувати

            execution_time = time.perf_counter() - start
            result = UpdateResult(
                operation="add_node",
                success=True,
                mst_edges=self.mst_edges,
                total_cost=self.total_cost,
                total_distance=self.total_distance,
                execution_time=execution_time,
                rebuild_required=False,
                message=f"Додано ізольовану вершину {node.name}"
            )
            self._history.append(result)
            return result

        # найдешевше ребро до нової вершини → одразу в МКД
        min_edge = min(new_edges, key=lambda e: e.weight)
        self._mst_edges.append(min_edge)
        self._mst_edge_set.add((min_edge.node1, min_edge.node2))

        execution_time = time.perf_counter() - start

        result = UpdateResult(
            operation="add_node",
            success=True,
            mst_edges=self.mst_edges,
            total_cost=self.total_cost,
            total_distance=self.total_distance,
            execution_time=execution_time,
            rebuild_required=False,
            affected_edges=[min_edge],
            message=f"Додано {node.name}, підключено через ребро вартістю {min_edge.weight:,.2f} грн"
        )

        self._history.append(result)
        return result

    def remove_node(self, node_id: int) -> UpdateResult:
        """Видаляє вершину і перебудовує МКД (інкрементально тут не вийде —
        видалене ребро могло бути єдиним мостом між компонентами)."""
        start = time.perf_counter()

        if not self.graph.has_node(node_id):
            return UpdateResult(
                operation="remove_node",
                success=False,
                mst_edges=self.mst_edges,
                total_cost=self.total_cost,
                total_distance=self.total_distance,
                execution_time=time.perf_counter() - start,
                rebuild_required=False,
                message=f"Вершина {node_id} не існує"
            )

        node_name = self.graph.get_node(node_id).name

        removed_edges = [e for e in self._mst_edges
                         if e.node1 == node_id or e.node2 == node_id]

        self.graph.remove_node(node_id)
        rebuild_time = self._rebuild()

        execution_time = time.perf_counter() - start

        result = UpdateResult(
            operation="remove_node",
            success=True,
            mst_edges=self.mst_edges,
            total_cost=self.total_cost,
            total_distance=self.total_distance,
            execution_time=execution_time,
            rebuild_required=True,
            affected_edges=removed_edges,
            message=f"Видалено {node_name}, перебудовано МКД"
        )

        self._history.append(result)
        return result

    def update_edge_cost(self,
                         node1: int,
                         node2: int,
                         new_cost_per_meter: float) -> UpdateResult:
        """Змінює вартість ребра і перебудовує МКД."""
        start = time.perf_counter()

        if not self.graph.has_edge(node1, node2):
            return UpdateResult(
                operation="update_edge",
                success=False,
                mst_edges=self.mst_edges,
                total_cost=self.total_cost,
                total_distance=self.total_distance,
                execution_time=time.perf_counter() - start,
                rebuild_required=False,
                message=f"Ребро ({node1}, {node2}) не існує"
            )

        old_edge = self.graph.get_edge(node1, node2)
        old_cost = old_edge.weight
        was_in_mst = self._is_in_mst(node1, node2)

        # пересоздаємо ребро з новою вартістю і перебудовуємо
        self.graph.remove_edge(node1, node2)
        new_edge = Edge(node1, node2, old_edge.distance, new_cost_per_meter)
        self.graph.add_edge(new_edge)
        self._rebuild()

        execution_time = time.perf_counter() - start

        is_in_mst_now = self._is_in_mst(node1, node2)

        # Формуємо повідомлення
        if was_in_mst and not is_in_mst_now:
            msg = f"Ребро ({node1}, {node2}) виключено з МКД (вартість: {old_cost:,.0f} → {new_edge.weight:,.0f})"
        elif not was_in_mst and is_in_mst_now:
            msg = f"Ребро ({node1}, {node2}) додано до МКД (вартість: {old_cost:,.0f} → {new_edge.weight:,.0f})"
        else:
            msg = f"Вартість ребра ({node1}, {node2}) змінено: {old_cost:,.0f} → {new_edge.weight:,.0f}"

        result = UpdateResult(
            operation="update_edge",
            success=True,
            mst_edges=self.mst_edges,
            total_cost=self.total_cost,
            total_distance=self.total_distance,
            execution_time=execution_time,
            rebuild_required=True,
            affected_edges=[new_edge],
            message=msg
        )

        self._history.append(result)
        return result

    def get_mst_result(self) -> MSTResult:
        """Повертає поточний стан МКД як MSTResult."""
        return MSTResult(
            edges=self.mst_edges,
            total_cost=self.total_cost,
            total_distance=self.total_distance,
            execution_time=0.0,
            steps=[],
            algorithm_name="Dynamic MST"
        )

    def print_status(self) -> None:
        """Виводить поточний стан МКД."""
        print(f"\n{'=' * 50}")
        print("ПОТОЧНИЙ СТАН МКД")
        print(f"{'=' * 50}")
        print(f"Вершин у графі: {self.graph.node_count()}")
        print(f"Ребер у графі: {self.graph.edge_count()}")
        print(f"Ребер у МКД: {len(self._mst_edges)}")
        print(f"Загальна довжина: {self.total_distance:,.2f} м")
        print(f"Загальна вартість: {self.total_cost:,.2f} грн")

        if self._mst_edges:
            print(f"\nРебра МКД:")
            for i, e in enumerate(self._mst_edges, 1):
                n1 = self.graph.get_node(e.node1).name
                n2 = self.graph.get_node(e.node2).name
                print(f"  {i}. {n1} — {n2}: {e.distance:,.0f} м, {e.weight:,.0f} грн")

    def print_history(self) -> None:
        """Виводить історію операцій."""
        print(f"\n{'=' * 50}")
        print("ІСТОРІЯ ОПЕРАЦІЙ")
        print(f"{'=' * 50}")

        if not self._history:
            print("Операцій не було")
            return

        for i, result in enumerate(self._history, 1):
            print(f"\n{i}. {result}")


# Приклад використання
if __name__ == "__main__":
    from models import Node, Graph

    print("=" * 60)
    print("ДЕМОНСТРАЦІЯ ДИНАМІЧНОГО МКД")
    print("=" * 60)

    # Початковий граф
    nodes = [
        Node(1, "Центральна", 48.4647, 35.0462),
        Node(2, "Північна", 48.5012, 35.0678),
        Node(3, "Південна", 48.4234, 35.0891),
        Node(4, "Східна", 48.4589, 35.1234),
    ]

    graph = Graph.from_nodes(nodes, cost_per_meter=150.0)

    # Створюємо динамічне МКД
    dynamic = DynamicMST(graph)
    dynamic.print_status()

    # Додаємо нову підстанцію
    print("\n" + "=" * 60)
    print("ОПЕРАЦІЯ: Додавання нової підстанції")
    print("=" * 60)

    new_node = Node(5, "Західна", 48.4701, 34.9876)
    result = dynamic.add_node(new_node, cost_per_meter=150.0)
    print(result)

    dynamic.print_status()

    # Видаляємо підстанцію
    print("\n" + "=" * 60)
    print("ОПЕРАЦІЯ: Видалення підстанції")
    print("=" * 60)

    result = dynamic.remove_node(3)
    print(result)

    dynamic.print_status()

    # Історія
    dynamic.print_history()