"""
Прім з indexed PQ і decrease-key.

Замість lazy deletion (як у звичайному Прімі) тримаємо на кожну вершину
рівно один запис у черзі і оновлюємо ключ через decrease_key. Купа не
роздувається до O(E), залишається ≤ V елементів.
"""

from typing import List, Set, Optional, Dict

from .base_mst import BaseMST
from models import Edge, Graph


class IndexedMinPQ:
    """Indexed min-heap. Елементи — ID вершин, у кожної свій ключ-пріоритет."""

    __slots__ = ('_heap', '_pos', '_key')

    def __init__(self):
        self._heap: List[int] = []
        self._pos: Dict[int, int] = {}
        self._key: Dict[int, float] = {}

    def __len__(self) -> int:
        return len(self._heap)

    def __bool__(self) -> bool:
        return bool(self._heap)

    def contains(self, vertex: int) -> bool:
        return vertex in self._pos

    def get_key(self, vertex: int) -> float:
        return self._key[vertex]

    def insert(self, vertex: int, key: float) -> None:
        if vertex in self._pos:
            raise ValueError(f"Вершина {vertex} вже у черзі")
        self._heap.append(vertex)
        idx = len(self._heap) - 1
        self._pos[vertex] = idx
        self._key[vertex] = key
        self._sift_up(idx)

    def decrease_key(self, vertex: int, new_key: float) -> None:
        if vertex not in self._pos:
            raise KeyError(f"Вершина {vertex} не у черзі")
        if new_key > self._key[vertex]:
            raise ValueError(
                f"decrease_key вимагає нового ключа <= поточного "
                f"({new_key} > {self._key[vertex]})"
            )
        self._key[vertex] = new_key
        self._sift_up(self._pos[vertex])

    def extract_min(self) -> int:
        if not self._heap:
            raise IndexError("extract_min з порожньої черги")
        top = self._heap[0]
        last = self._heap.pop()
        del self._pos[top]
        del self._key[top]
        if self._heap:
            self._heap[0] = last
            self._pos[last] = 0
            self._sift_down(0)
        return top

    def _sift_up(self, i: int) -> None:
        heap = self._heap
        pos = self._pos
        key = self._key
        while i > 0:
            parent = (i - 1) // 2
            if key[heap[i]] < key[heap[parent]]:
                # Свопаємо у масиві та оновлюємо позиції
                heap[i], heap[parent] = heap[parent], heap[i]
                pos[heap[i]] = i
                pos[heap[parent]] = parent
                i = parent
            else:
                return

    def _sift_down(self, i: int) -> None:
        heap = self._heap
        pos = self._pos
        key = self._key
        n = len(heap)
        while True:
            left = 2 * i + 1
            right = 2 * i + 2
            smallest = i
            if left < n and key[heap[left]] < key[heap[smallest]]:
                smallest = left
            if right < n and key[heap[right]] < key[heap[smallest]]:
                smallest = right
            if smallest == i:
                return
            heap[i], heap[smallest] = heap[smallest], heap[i]
            pos[heap[i]] = i
            pos[heap[smallest]] = smallest
            i = smallest


class PrimMSTIndexed(BaseMST):
    """Прім з indexed PQ. Витягуємо найдешевшу вершину, релаксуємо її сусідів."""

    @property
    def algorithm_name(self) -> str:
        return "Prim (Indexed PQ, decrease-key)"

    def _find_mst_impl(self, start_node: Optional[int] = None) -> List[Edge]:
        if self.graph.node_count() <= 1:
            return []

        if start_node is None:
            start_node = self.graph.get_node_ids()[0]

        mst_edges: List[Edge] = []
        in_mst: Set[int] = set()
        edge_to: Dict[int, Optional[Edge]] = {}
        target_nodes = self.graph.node_count()

        # Для незв'язних графів — рестарт з невідвіданої вершини
        while len(in_mst) < target_nodes:
            if start_node in in_mst:
                for nid in self.graph.get_node_ids():
                    if nid not in in_mst:
                        start_node = nid
                        break

            pq = IndexedMinPQ()
            pq.insert(start_node, 0.0)
            edge_to[start_node] = None

            while pq:
                v = pq.extract_min()
                in_mst.add(v)

                e = edge_to.get(v)
                if e is not None:
                    mst_edges.append(e)
                    self._record_step(mst_edges)

                for edge in self.graph.get_neighbors(v):
                    u = edge.get_other_node(v)
                    if u in in_mst:
                        continue
                    w = edge.weight
                    if pq.contains(u):
                        if w < pq.get_key(u):
                            edge_to[u] = edge
                            pq.decrease_key(u, w)
                    else:
                        edge_to[u] = edge
                        pq.insert(u, w)

        return mst_edges
