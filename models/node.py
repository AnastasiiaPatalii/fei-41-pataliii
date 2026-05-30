"""Підстанція = вершина графа. x — широта, y — довгота."""

from dataclasses import dataclass, field
from typing import Optional
import math

EARTH_RADIUS_M = 6_371_000  # м


@dataclass
class Node:
    id: int
    name: str
    x: float  # широта
    y: float  # довгота
    metadata: Optional[dict] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.id, int) or self.id < 0:
            raise ValueError(f"ID має бути невід'ємним цілим числом, отримано: {self.id}")

        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Назва підстанції не може бути порожньою")

        if not (-90 <= self.x <= 90):
            raise ValueError(f"Широта має бути в межах [-90, 90], отримано: {self.x}")

        if not (-180 <= self.y <= 180):
            raise ValueError(f"Довгота має бути в межах [-180, 180], отримано: {self.y}")

    def distance_to(self, other: 'Node', method: str = 'haversine') -> float:
        """method='haversine' (м, для GPS) або 'euclidean' (в одиницях координат)."""
        if method == 'haversine':
            return self._haversine_distance(other)
        elif method == 'euclidean':
            return self._euclidean_distance(other)
        else:
            raise ValueError(f"Невідомий метод обчислення відстані: {method}. "
                             f"Доступні: 'haversine', 'euclidean'")

    def _haversine_distance(self, other: 'Node') -> float:
        """Гаверсинус (м) — враховує кривизну Землі."""
        lat1 = math.radians(self.x)
        lat2 = math.radians(other.x)
        lon1 = math.radians(self.y)
        lon2 = math.radians(other.y)

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = math.sin(dlat / 2) ** 2 + \
            math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))

        return EARTH_RADIUS_M * c

    def _euclidean_distance(self, other: 'Node') -> float:
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def to_dict(self) -> dict:
        return {
            'id': self.id,
            'name': self.name,
            'x': self.x,
            'y': self.y,
            'metadata': self.metadata
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'Node':
        return cls(
            id=data['id'],
            name=data['name'],
            x=data['x'],
            y=data['y'],
            metadata=data.get('metadata', {})
        )

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Node):
            return NotImplemented
        return self.id == other.id

    def __repr__(self) -> str:
        return f"Node(id={self.id}, name='{self.name}', x={self.x}, y={self.y})"

    def __str__(self) -> str:
        return f"{self.name} (ID: {self.id})"


# Приклад використання (для тестування)
if __name__ == "__main__":
    # Створення підстанцій у Дніпрі
    central = Node(1, "Підстанція-Центральна", 48.4647, 35.0462)
    north = Node(2, "Підстанція-Північна", 48.5012, 35.0678)
    south = Node(3, "Підстанція-Південна", 48.4234, 35.0891)

    # Обчислення відстаней
    print(f"Відстань {central} → {north}:")
    print(f"  Гаверсинус: {central.distance_to(north, 'haversine'):.2f} м")
    print(f"  Евклідова:  {central.distance_to(north, 'euclidean'):.6f}")

    print(f"\nВідстань {central} → {south}:")
    print(f"  Гаверсинус: {central.distance_to(south, 'haversine'):.2f} м")

    # Серіалізація
    print(f"\nСеріалізація: {central.to_dict()}")

    # Десеріалізація
    restored = Node.from_dict(central.to_dict())
    print(f"Відновлено: {restored}")