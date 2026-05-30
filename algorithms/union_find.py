"""Disjoint Set Union (Union-Find) для Крускала. Зі стисненням шляху і
об'єднанням за рангом — амортизовано O(α(n)) на операцію."""

from typing import Dict, List, Any, Optional


class UnionFind:
    """Класична DSU."""

    def __init__(self, elements):
        self.parent: Dict[Any, Any] = {x: x for x in elements}
        self.rank: Dict[Any, int] = {x: 0 for x in elements}
        self._num_components = len(self.parent)

    def find(self, x) -> Any:
        """Корінь множини з path compression.

        Raises:
            KeyError: Якщо елемент не існує
        """
        if x not in self.parent:
            raise KeyError(f"Елемент {x} не знайдено в Union-Find")

        # Стиснення шляху (рекурсивне)
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])

        return self.parent[x]

    def union(self, x, y) -> bool:
        """True якщо множини були різні і ми їх об'єднали; False якщо вже разом."""
        root_x = self.find(x)
        root_y = self.find(y)

        if root_x == root_y:
            return False

        # union by rank — менше дерево під більше
        if self.rank[root_x] < self.rank[root_y]:
            root_x, root_y = root_y, root_x

        self.parent[root_y] = root_x

        if self.rank[root_x] == self.rank[root_y]:
            self.rank[root_x] += 1

        self._num_components -= 1
        return True

    def connected(self, x, y) -> bool:
        return self.find(x) == self.find(y)

    @property
    def num_components(self) -> int:
        """Повертає кількість незалежних множин (компонент)."""
        return self._num_components

    def get_component(self, x) -> List[Any]:
        """
        Повертає всі елементи множини, до якої належить x.

        Args:
            x: Елемент множини

        Returns:
            Список всіх елементів цієї множини
        """
        root = self.find(x)
        return [elem for elem in self.parent if self.find(elem) == root]

    def __len__(self) -> int:
        """Повертає загальну кількість елементів."""
        return len(self.parent)

    def __contains__(self, x) -> bool:
        """Перевіряє, чи елемент є в структурі."""
        return x in self.parent

    def __repr__(self) -> str:
        return f"UnionFind(elements={len(self)}, components={self.num_components})"


# Приклад використання
if __name__ == "__main__":
    print("=== Тест Union-Find ===\n")

    # Створюємо структуру для 6 елементів
    uf = UnionFind([0, 1, 2, 3, 4, 5])
    print(f"Початково: {uf}")

    # Об'єднуємо
    operations = [(0, 1), (2, 3), (4, 5), (0, 2), (3, 5)]

    for x, y in operations:
        result = uf.union(x, y)
        status = "об'єднано" if result else "вже разом"
        print(f"union({x}, {y}): {status}, компонент: {uf.num_components}")

    print(f"\nПеревірка зв'язності:")
    print(f"  connected(0, 5): {uf.connected(0, 5)}")
    print(f"  connected(1, 4): {uf.connected(1, 4)}")

    print("\nUnion-Find працює!")