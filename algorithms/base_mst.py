"""BaseMST і MSTResult. Конкретні алгоритми перевизначають _find_mst_impl()."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional
import time
import tracemalloc
import warnings

from models import Edge, Graph


@dataclass
class MSTResult:
    """Результат МКД. peak_memory_bytes/current_memory_bytes — лише якщо
    find_mst() викликали з track_memory=True."""
    edges: List[Edge]
    total_cost: float
    total_distance: float
    execution_time: float
    steps: List[List[Edge]] = field(default_factory=list)
    algorithm_name: str = ""
    peak_memory_bytes: Optional[int] = None
    current_memory_bytes: Optional[int] = None

    @property
    def peak_memory_kb(self) -> float:
        return (self.peak_memory_bytes or 0) / 1024

    @property
    def peak_memory_mb(self) -> float:
        return (self.peak_memory_bytes or 0) / (1024 * 1024)

    @property
    def edge_count(self) -> int:
        """Кількість ребер у МКД."""
        return len(self.edges)

    def to_dict(self) -> dict:
        """Серіалізація результату для експорту в JSON."""
        return {
            'algorithm': self.algorithm_name,
            'execution_time_ms': self.execution_time * 1000,
            'peak_memory_kb': self.peak_memory_kb if self.peak_memory_bytes else None,
            'total_cost': self.total_cost,
            'total_distance_m': self.total_distance,
            'num_edges': self.edge_count,
            'edges': [
                {
                    'node1': e.node1,
                    'node2': e.node2,
                    'distance': e.distance,
                    'weight': e.weight
                }
                for e in self.edges
            ]
        }

    def __str__(self) -> str:
        """Читабельне представлення результату."""
        mem_str = ""
        if self.peak_memory_bytes is not None:
            mem_str = f"\n  Пам'ять (пік): {self.peak_memory_kb:,.1f} КБ"
        return (
            f"МКД ({self.algorithm_name}):\n"
            f"  Ребер: {self.edge_count}\n"
            f"  Довжина: {self.total_distance:,.2f} м\n"
            f"  Вартість: {self.total_cost:,.2f} грн\n"
            f"  Час: {self.execution_time * 1000:.3f} мс"
            f"{mem_str}"
        )


class BaseMST(ABC):
    """Базовий клас МКД-алгоритмів. Виміри часу/пам'яті і збереження кроків —
    тут; сам алгоритм — у _find_mst_impl()."""

    def __init__(self, graph: Graph, record_steps: bool = True):
        """record_steps=False для бенчмарків (інакше зайва пам'ять на історії)."""
        self.graph = graph
        self.record_steps = record_steps
        self._steps: List[List[Edge]] = []

    @property
    @abstractmethod
    def algorithm_name(self) -> str:
        """Назва алгоритму для звітів."""
        pass

    @abstractmethod
    def _find_mst_impl(self, start_node: Optional[int] = None) -> List[Edge]:
        """
        Внутрішня реалізація алгоритму.

        Args:
            start_node: Початкова вершина (для Пріма), None для Крускала

        Returns:
            Список ребер МКД
        """
        pass

    def find_mst(self, start_node: Optional[int] = None,
                 track_memory: bool = False) -> MSTResult:
        """Будує МКД. track_memory=True вмикає tracemalloc (оверхед ~5-10× по часу,
        тому для чистого вимірювання часу — False)."""
        self._steps.clear()

        if self.graph.node_count() == 0:
            return MSTResult(
                edges=[],
                total_cost=0.0,
                total_distance=0.0,
                execution_time=0.0,
                steps=[],
                algorithm_name=self.algorithm_name
            )

        peak_mem: Optional[int] = None
        current_mem: Optional[int] = None

        if track_memory:
            was_tracing = tracemalloc.is_tracing()
            if not was_tracing:
                tracemalloc.start()
            else:
                tracemalloc.clear_traces()

            start_time = time.perf_counter()
            mst_edges = self._find_mst_impl(start_node)
            execution_time = time.perf_counter() - start_time

            current_mem, peak_mem = tracemalloc.get_traced_memory()
            if not was_tracing:
                tracemalloc.stop()
        else:
            start_time = time.perf_counter()
            mst_edges = self._find_mst_impl(start_node)
            execution_time = time.perf_counter() - start_time

        # незв'язний граф → отримаємо кістяковий ліс
        expected_edges = self.graph.node_count() - 1
        if len(mst_edges) < expected_edges and self.graph.node_count() > 1:
            warnings.warn(
                f"Граф незв'язний: МКД містить {len(mst_edges)} ребер "
                f"замість очікуваних {expected_edges}. "
                f"Результат є мінімальним кістяковим лісом.",
                RuntimeWarning
            )

        total_cost = sum(e.weight for e in mst_edges)
        total_distance = sum(e.distance for e in mst_edges)

        return MSTResult(
            edges=mst_edges,
            total_cost=total_cost,
            total_distance=total_distance,
            execution_time=execution_time,
            steps=self._steps.copy(),
            algorithm_name=self.algorithm_name,
            peak_memory_bytes=peak_mem,
            current_memory_bytes=current_mem
        )

    def _record_step(self, current_edges: List[Edge]) -> None:
        """
        Зберігає поточний стан для покрокової візуалізації.

        Args:
            current_edges: Поточний список ребер МКД
        """
        if self.record_steps:
            self._steps.append(list(current_edges))