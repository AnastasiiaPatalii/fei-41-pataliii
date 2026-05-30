"""Лінія електропередачі = ребро графа. weight = distance * cost_per_meter."""

from dataclasses import dataclass, field
from typing import Optional, Tuple
import itertools
import math

_id_counter = itertools.count()  # для tie-break у купі на однакових вагах


@dataclass
class Edge:
    """Неорієнтоване ребро. (node1, node2) eq (node2, node1) після нормалізації."""

    node1: int
    node2: int
    distance: float
    cost_per_meter: float = 100.0
    metadata: Optional[dict] = field(default_factory=dict)

    # для стабільного порівняння на однакових вагах
    _id: int = field(default=0, init=False, repr=False, compare=False)

    def __post_init__(self):
        if not isinstance(self.node1, int) or self.node1 < 0:
            raise ValueError(f"node1 має бути невід'ємним цілим числом, отримано: {self.node1}")

        if not isinstance(self.node2, int) or self.node2 < 0:
            raise ValueError(f"node2 має бути невід'ємним цілим числом, отримано: {self.node2}")

        if self.node1 == self.node2:
            raise ValueError(f"Петля заборонена: node1 == node2 == {self.node1}")

        if self.distance <= 0:
            raise ValueError(f"Довжина має бути додатною, отримано: {self.distance}")

        if self.cost_per_meter < 0:
            raise ValueError(f"Вартість за метр не може бути від'ємною, отримано: {self.cost_per_meter}")

        # завжди node1 < node2
        if self.node1 > self.node2:
            self.node1, self.node2 = self.node2, self.node1

        self._id = next(_id_counter)

    @property
    def weight(self) -> float:
        return self.distance * self.cost_per_meter

    @property
    def nodes(self) -> Tuple[int, int]:
        return (self.node1, self.node2)

    def contains_node(self, node_id: int) -> bool:
        return node_id == self.node1 or node_id == self.node2

    def get_other_node(self, node_id: int) -> int:
        if node_id == self.node1:
            return self.node2
        elif node_id == self.node2:
            return self.node1
        else:
            raise ValueError(f"Вершина {node_id} не належить ребру ({self.node1}, {self.node2})")

    def to_dict(self) -> dict:
        return {
            'node1': self.node1,
            'node2': self.node2,
            'distance': self.distance,
            'cost_per_meter': self.cost_per_meter,
            'weight': self.weight,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Edge':
        """Створює ребро зі словника."""
        return cls(
            node1=data['node1'],
            node2=data['node2'],
            distance=data['distance'],
            cost_per_meter=data.get('cost_per_meter', 100.0),
            metadata=data.get('metadata', {})
        )

    @classmethod
    def from_nodes(cls, node1: 'Node', node2: 'Node',
                   cost_per_meter: float = 100.0,
                   distance_method: str = 'haversine') -> 'Edge':
        """Створює ребро з двох об'єктів Node, автоматично обчислюючи відстань."""
        from .node import Node
        distance = node1.distance_to(node2, method=distance_method)
        return cls(
            node1=node1.id,
            node2=node2.id,
            distance=distance,
            cost_per_meter=cost_per_meter
        )

    def __lt__(self, other: 'Edge') -> bool:
        """За вагою + _id як tie-break (потрібно для heapq на рівних вагах)."""
        if not isinstance(other, Edge):
            return NotImplemented
        if self.weight != other.weight:
            return self.weight < other.weight
        return self._id < other._id

    def __le__(self, other: 'Edge') -> bool:
        if not isinstance(other, Edge):
            return NotImplemented
        return self.weight <= other.weight

    def __gt__(self, other: 'Edge') -> bool:
        if not isinstance(other, Edge):
            return NotImplemented
        return self.weight > other.weight

    def __ge__(self, other: 'Edge') -> bool:
        if not isinstance(other, Edge):
            return NotImplemented
        return self.weight >= other.weight

    def __hash__(self) -> int:
        return hash((self.node1, self.node2))

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Edge):
            return NotImplemented
        return self.node1 == other.node1 and self.node2 == other.node2

    def __repr__(self) -> str:
        return (f"Edge(node1={self.node1}, node2={self.node2}, "
                f"distance={self.distance:.2f}, cost_per_meter={self.cost_per_meter}, "
                f"weight={self.weight:.2f})")

    def __str__(self) -> str:
        return f"({self.node1} ↔ {self.node2}): {self.distance:.2f} м, вартість: {self.weight:,.2f} грн"