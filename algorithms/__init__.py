"""
Модуль algorithms — Алгоритми побудови МКД

Експортує:
    - BaseMST: Базовий клас алгоритму
    - MSTResult: Результат побудови МКД
    - PrimMST: Алгоритм Пріма (бінарна купа, lazy deletion) — поточний
    - PrimMSTDAry: Прім з d-арною купою (параметризована)
    - PrimMSTIndexed: Канонічний Прім з indexed PQ та decrease-key
    - KruskalMST: Алгоритм Крускала (Union-Find)
    - UnionFind: Структура даних для неперетинних множин
    - DynamicMST: Динамічне оновлення МКД
    - UpdateResult: Результат операції оновлення
"""

from .base_mst import BaseMST, MSTResult
from .prim import PrimMST
from .prim_dary import PrimMSTDAry
from .prim_indexed import PrimMSTIndexed
from .union_find import UnionFind
from .kruskal import KruskalMST
from .dynamic_mst import DynamicMST, UpdateResult
from .bridges import find_bridges, is_2_edge_connected, count_bridges
from .augmentation import augment_to_2_edge_connected, AugmentationResult

__all__ = [
    'BaseMST',
    'MSTResult',
    'PrimMST',
    'PrimMSTDAry',
    'PrimMSTIndexed',
    'KruskalMST',
    'UnionFind',
    'DynamicMST',
    'UpdateResult',
    # 2-edge-connectivity
    'find_bridges',
    'is_2_edge_connected',
    'count_bridges',
    'augment_to_2_edge_connected',
    'AugmentationResult',
]