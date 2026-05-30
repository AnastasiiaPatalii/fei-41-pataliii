"""
Прім з d-арною купою. Параметр d задається у конструкторі.

heapq у стандартній бібліотеці бінарний і не параметризується,
тому купу довелось писати окремо.
"""

from typing import List, Set, Optional, Tuple

from .base_mst import BaseMST
from models import Edge, Graph


class DAryHeap:
    """Min-heap з d дітьми у вузла. Елементи — (priority, edge)."""

    __slots__ = ('_data', '_d')

    def __init__(self, d: int = 4):
        if d < 2:
            raise ValueError(f"d має бути >= 2, отримано: {d}")
        self._d = d
        self._data: List[Tuple[float, Edge]] = []

    def __len__(self) -> int:
        return len(self._data)

    def __bool__(self) -> bool:
        return bool(self._data)

    def push(self, item: Tuple[float, Edge]) -> None:
        self._data.append(item)
        self._sift_up(len(self._data) - 1)

    def pop(self) -> Tuple[float, Edge]:
        if not self._data:
            raise IndexError("pop from empty heap")
        top = self._data[0]
        last = self._data.pop()
        if self._data:
            self._data[0] = last
            self._sift_down(0)
        return top

    def _sift_up(self, i: int) -> None:
        d = self._d
        data = self._data
        while i > 0:
            parent = (i - 1) // d
            if data[i] < data[parent]:
                data[i], data[parent] = data[parent], data[i]
                i = parent
            else:
                break

    def _sift_down(self, i: int) -> None:
        d = self._d
        data = self._data
        n = len(data)
        while True:
            first_child = d * i + 1
            if first_child >= n:
                return
            # Знаходимо найменшу з d дітей
            min_child = first_child
            last_child = min(first_child + d, n)
            for c in range(first_child + 1, last_child):
                if data[c] < data[min_child]:
                    min_child = c
            if data[min_child] < data[i]:
                data[i], data[min_child] = data[min_child], data[i]
                i = min_child
            else:
                return


class PrimMSTDAry(BaseMST):
    """Прім з d-арною купою. Логіка та сама, що у PrimMST, але купа — своя."""

    def __init__(self, graph: Graph, record_steps: bool = True, d: int = 4):
        super().__init__(graph, record_steps)
        if d < 2:
            raise ValueError(f"d має бути >= 2, отримано: {d}")
        self.d = d

    @property
    def algorithm_name(self) -> str:
        return f"Prim ({self.d}-ary Heap)"

    def _find_mst_impl(self, start_node: Optional[int] = None) -> List[Edge]:
        if self.graph.node_count() <= 1:
            return []

        if start_node is None:
            start_node = self.graph.get_node_ids()[0]

        mst_edges: List[Edge] = []
        visited: Set[int] = set()
        target_nodes = self.graph.node_count()

        while len(visited) < target_nodes:
            if start_node in visited:
                for node_id in self.graph.get_node_ids():
                    if node_id not in visited:
                        start_node = node_id
                        break

            visited.add(start_node)
            heap = DAryHeap(d=self.d)

            for edge in self.graph.get_neighbors(start_node):
                heap.push((edge.weight, edge))

            while heap:
                weight, edge = heap.pop()
                n1_visited = edge.node1 in visited
                n2_visited = edge.node2 in visited

                if n1_visited and n2_visited:
                    continue  # обидва кінці вже у дереві — викидаємо

                new_node = edge.node2 if n1_visited else edge.node1
                visited.add(new_node)
                mst_edges.append(edge)
                self._record_step(mst_edges)

                for next_edge in self.graph.get_neighbors(new_node):
                    other = next_edge.get_other_node(new_node)
                    if other not in visited:
                        heap.push((next_edge.weight, next_edge))

        return mst_edges
