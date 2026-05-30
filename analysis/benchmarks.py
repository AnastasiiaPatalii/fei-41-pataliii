"""
Бенчмарки алгоритмів МКД.

Методологія: time.perf_counter() з warm-up (перші 3 прогони ігноруються),
вимкнений gc, адаптивна к-сть повторів (ціль ~1 сек на точку),
IQR-фільтрація викидів, 95% CI через t-розподіл Стьюдента.
При CV > 15% видається попередження.
"""

import gc
import time
import math
import random
import tracemalloc
import warnings
import statistics
from dataclasses import dataclass, field
from typing import List, Type, Optional, Dict, Tuple

from models import Graph, Node, Edge
from algorithms import (
    BaseMST, PrimMST, PrimMSTDAry, PrimMSTIndexed, KruskalMST, DynamicMST
)
from data.generator import GraphGenerator

# КОНФІГУРАЦІЯ


@dataclass
class BenchmarkConfig:
    """
    Параметри вимірювання.

    Attributes:
        warmup_iterations: Кількість «прогрівних» запусків (не враховуються)
        min_iterations: Мінімальна кількість вимірювань
        max_iterations: Максимальна кількість вимірювань
        target_time: Бажаний загальний час вимірювання (секунди)
        confidence_level: Рівень довіри для CI (0.95 = 95%)
        cv_warning_threshold: Поріг CV для попередження про нестабільність
    """
    warmup_iterations: int = 3
    min_iterations: int = 7
    max_iterations: int = 200
    target_time: float = 1.0
    confidence_level: float = 0.95
    cv_warning_threshold: float = 0.15

    def calculate_iterations(self, single_run_time: float) -> int:
        """
        Адаптивна кількість ітерацій на основі часу одного запуску.
        Логіка: витратити ~target_time секунд на кожну точку.
        """
        if single_run_time <= 0:
            return self.max_iterations
        estimated = int(self.target_time / single_run_time)
        return max(self.min_iterations, min(estimated, self.max_iterations))


# РЕЗУЛЬТАТ ВИМІРЮВАННЯ

# Таблиця критичних значень t-розподілу Стьюдента (двосторонній, α=0.05)
_T_TABLE_95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    15: 2.131, 20: 2.086, 25: 2.060, 30: 2.042, 50: 2.009,
    100: 1.984, 200: 1.972,
}


def _get_t_value(df: int) -> float:
    """Повертає t-значення для df ступенів свободи (95% CI)."""
    if df <= 0:
        return 0.0
    keys = sorted(_T_TABLE_95.keys())
    for k in keys:
        if df <= k:
            return _T_TABLE_95[k]
    return 1.96  # Для великих df → нормальний розподіл


@dataclass
class BenchmarkResult:
    """
    Результат вимірювання продуктивності алгоритму.

    Зберігає повну статистичну інформацію для аналізу в п.3.4.2.
    Поля пам'яті заповнюються через measure_full() або measure_memory().
    """
    algorithm_name: str
    node_count: int
    edge_count: int
    density: float
    iterations: int
    iterations_after_filter: int
    mean_time: float
    std_time: float
    median_time: float
    min_time: float
    max_time: float
    ci_lower: float
    ci_upper: float
    cv: float
    mst_cost: float
    peak_memory_kb: float = 0.0
    peak_memory_std_kb: float = 0.0
    memory_iterations: int = 0

    @property
    def mean_time_ms(self) -> float:
        """Середній час у мілісекундах."""
        return self.mean_time * 1000

    @property
    def std_time_ms(self) -> float:
        """Стандартне відхилення у мілісекундах."""
        return self.std_time * 1000

    @property
    def ci_lower_ms(self) -> float:
        return self.ci_lower * 1000

    @property
    def ci_upper_ms(self) -> float:
        return self.ci_upper * 1000

    @property
    def peak_memory_mb(self) -> float:
        return self.peak_memory_kb / 1024

    def __str__(self) -> str:
        mem_str = ""
        if self.peak_memory_kb > 0:
            mem_str = (f"\n  Пам'ять (пік): {self.peak_memory_kb:,.1f} ± "
                       f"{self.peak_memory_std_kb:,.1f} КБ "
                       f"(n={self.memory_iterations})")
        return (
            f"{self.algorithm_name}:\n"
            f"  Граф: V={self.node_count}, E={self.edge_count} "
            f"(Щільність: {self.density:.1%})\n"
            f"  Час: {self.mean_time_ms:.3f} ± {self.std_time_ms:.3f} мс "
            f"(n={self.iterations_after_filter})\n"
            f"  95% CI: [{self.ci_lower_ms:.3f}, {self.ci_upper_ms:.3f}] мс\n"
            f"  Медіана: {self.median_time * 1000:.3f} мс, "
            f"CV: {self.cv:.1%}"
            f"{mem_str}\n"
            f"  Вартість МКД: {self.mst_cost:,.2f} грн"
        )


# ДОПОМІЖНІ ФУНКЦІЇ

def _iqr_filter(data: List[float]) -> List[float]:
    """
    Фільтрація викидів методом міжквартильного розмаху (IQR).

    Стандартна статистична методика:
        Q1 = 25-й перцентиль, Q3 = 75-й перцентиль
        IQR = Q3 - Q1
        Допустимий діапазон: [Q1 - 1.5·IQR, Q3 + 1.5·IQR]

    Видаляє аномальні заміри, спричинені GC, перемиканням контексту ОС тощо.
    """
    if len(data) < 4:
        return data

    sorted_data = sorted(data)
    n = len(sorted_data)
    q1 = sorted_data[n // 4]
    q3 = sorted_data[3 * n // 4]
    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    filtered = [x for x in data if lower_bound <= x <= upper_bound]

    # Якщо фільтр відкинув > 30% даних — щось не так, повертаємо все
    if len(filtered) < 0.7 * len(data):
        return data

    return filtered


def _compute_stats(times: List[float]) -> dict:
    """
    Обчислює повний набір статистик для масиву замірів.

    Returns:
        dict з ключами: mean, std, median, min, max, ci_lower, ci_upper, cv
    """
    n = len(times)
    mean = statistics.mean(times)
    std = statistics.stdev(times) if n > 1 else 0.0
    median = statistics.median(times)

    # 95% довірчий інтервал: mean ± t * (std / sqrt(n))
    if n > 1 and std > 0:
        t_val = _get_t_value(n - 1)
        margin = t_val * std / math.sqrt(n)
    else:
        margin = 0.0

    cv = std / mean if mean > 0 else 0.0

    return {
        'mean': mean,
        'std': std,
        'median': median,
        'min': min(times),
        'max': max(times),
        'ci_lower': mean - margin,
        'ci_upper': mean + margin,
        'cv': cv,
    }


def _compute_density(graph: Graph) -> float:
    """Щільність графа: E / (V*(V-1)/2). Для K_n = 1.0."""
    v = graph.node_count()
    max_edges = v * (v - 1) / 2 if v > 1 else 1
    return graph.edge_count() / max_edges


# ═══════════════════════════════════════════════════════════════
# ОСНОВНИЙ КЛАС БЕНЧМАРКІВ
# ═══════════════════════════════════════════════════════════════

class Benchmark:
    """
    Клас для вимірювання продуктивності алгоритмів МКД.

    Використання:
        bench = Benchmark()
        result = bench.measure(PrimMST, graph)
        print(result)
    """

    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or BenchmarkConfig()

    def measure(self, algorithm_class: Type[BaseMST],
                graph: Graph, verbose: bool = False) -> BenchmarkResult:
        """
        Вимірює час роботи алгоритму з повною статистичною обробкою.

        Кроки:
            1. Warm-up (прогрів кешу CPU та інтерпретатора)
            2. Пробний запуск -> визначення кількості ітерацій
            3. Основні вимірювання (GC вимкнений)
            4. IQR-фільтрація викидів
            5. Обчислення статистик та 95% CI
            6. Перевірка CV (попередження при > 15%)
        """
        algo_name = algorithm_class(graph, record_steps=False).algorithm_name

        # -- Крок 1: Warm-up --
        for _ in range(self.config.warmup_iterations):
            algorithm_class(graph, record_steps=False).find_mst()

        # -- Крок 2: Пробний запуск -> кількість ітерацій --
        start = time.perf_counter()
        reference_result = algorithm_class(graph, record_steps=False).find_mst()
        single_run = time.perf_counter() - start

        iterations = self.config.calculate_iterations(single_run)

        if verbose:
            print(f"  {algo_name}: ~{single_run*1000:.2f} мс/запуск -> "
                  f"{iterations} ітерацій")

        # -- Крок 3: Вимірювання (GC вимкнений) --
        raw_times = []
        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            for _ in range(iterations):
                algo = algorithm_class(graph, record_steps=False)
                start = time.perf_counter()
                algo.find_mst()
                elapsed = time.perf_counter() - start
                raw_times.append(elapsed)
        finally:
            if gc_was_enabled:
                gc.enable()

        # -- Крок 4: IQR-фільтрація --
        filtered_times = _iqr_filter(raw_times)

        # -- Крок 5: Статистики --
        stats = _compute_stats(filtered_times)

        # -- Крок 6: Перевірка CV --
        if stats['cv'] > self.config.cv_warning_threshold:
            warnings.warn(
                f"Нестабільні заміри для {algo_name} (V={graph.node_count()}): "
                f"CV={stats['cv']:.1%} > {self.config.cv_warning_threshold:.0%}. "
                f"Можливо, фоновий процес ОС заважає вимірюванням.",
                RuntimeWarning
            )

        return BenchmarkResult(
            algorithm_name=algo_name,
            node_count=graph.node_count(),
            edge_count=graph.edge_count(),
            density=_compute_density(graph),
            iterations=iterations,
            iterations_after_filter=len(filtered_times),
            mean_time=stats['mean'],
            std_time=stats['std'],
            median_time=stats['median'],
            min_time=stats['min'],
            max_time=stats['max'],
            ci_lower=stats['ci_lower'],
            ci_upper=stats['ci_upper'],
            cv=stats['cv'],
            mst_cost=reference_result.total_cost
        )

    # --- Пам'ять (окремо від часу, бо tracemalloc сповільнює) ---

    def measure_memory(self, algorithm_class: Type[BaseMST],
                       graph: Graph, iterations: int = 10) -> Dict[str, float]:
        """Пік пам'яті алгоритму через tracemalloc, усереднено по iterations прогонів."""
        peaks = []
        currents = []

        for _ in range(iterations):
            gc.collect()
            tracemalloc.start()
            try:
                algo = algorithm_class(graph, record_steps=False)
                algo.find_mst()
                current, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
            peaks.append(peak)
            currents.append(current)

        peak_mean = statistics.mean(peaks)
        peak_std = statistics.stdev(peaks) if len(peaks) > 1 else 0.0
        current_mean = statistics.mean(currents)

        return {
            'peak_mean_bytes': peak_mean,
            'peak_std_bytes': peak_std,
            'current_mean_bytes': current_mean,
            'iterations': iterations,
        }

    def measure_full(self, algorithm_class: Type[BaseMST],
                     graph: Graph, verbose: bool = False,
                     memory_iterations: int = 10) -> BenchmarkResult:
        """measure() + measure_memory() в один BenchmarkResult."""
        result = self.measure(algorithm_class, graph, verbose=verbose)
        mem = self.measure_memory(algorithm_class, graph, iterations=memory_iterations)
        result.peak_memory_kb = mem['peak_mean_bytes'] / 1024
        result.peak_memory_std_kb = mem['peak_std_bytes'] / 1024
        result.memory_iterations = mem['iterations']
        if verbose:
            print(f"    Пам'ять (пік): {result.peak_memory_kb:,.1f} ± "
                  f"{result.peak_memory_std_kb:,.1f} КБ "
                  f"(n={memory_iterations})")
        return result


    def full_comparison(self, sizes: List[int],
                        graph_type: str = 'random',
                        verbose: bool = True) -> Dict[str, List[BenchmarkResult]]:
        """
        Порівнює Пріма і Крускала на графах різного розміру.

        Args:
            sizes: список кількостей вершин [50, 100, 200, ...]
            graph_type: 'random' (повний), 'grid' (розріджений), 'cluster'
            verbose: виводити прогрес

        Returns:
            {'PrimMST': [...], 'KruskalMST': [...]}
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"  Аналіз складності | Тип графа: {graph_type}")
            print(f"  Розміри: {sizes}")
            print(f"{'='*60}")

        prim_results = []
        kruskal_results = []

        for n in sizes:
            gen = GraphGenerator(seed=42 + n)

            if graph_type == 'random':
                g = gen.random_graph(n, complete=True)
            elif graph_type == 'grid':
                rows = max(1, int(n ** 0.5))
                cols = max(1, n // rows)
                g = gen.grid_graph(rows, cols)
            else:
                num_clusters = max(2, n // 10)
                nodes_per = max(2, n // num_clusters)
                g = gen.cluster_graph(num_clusters, nodes_per)

            if verbose:
                print(f"\n  V={g.node_count()}, E={g.edge_count()} "
                      f"(щільність: {_compute_density(g):.1%})")

            p = self.measure_full(PrimMST, g, verbose=verbose)
            k = self.measure_full(KruskalMST, g, verbose=verbose)
            prim_results.append(p)
            kruskal_results.append(k)

            if verbose:
                winner = "Пріма" if p.mean_time < k.mean_time else "Крускала"
                ratio = max(p.mean_time, k.mean_time) / min(p.mean_time, k.mean_time)
                print(f"  -> Переможець: {winner} (у {ratio:.2f}x)")

        return {'PrimMST': prim_results, 'KruskalMST': kruskal_results}

    # ПОРІВНЯННЯ ВАРІАНТІВ ПРІМА

    def compare_prim_variants(self, sizes: List[int],
                              graph_type: str = 'random',
                              dary_d: int = 4,
                              verbose: bool = True
                              ) -> Dict[str, List[BenchmarkResult]]:
        """Усі 3 варіанти Пріма + Крускал на одних і тих самих графах."""
        if verbose:
            print(f"\n  Порівняння варіантів Пріма | Тип графа: {graph_type}")
            print(f"  Розміри: {sizes} | d для d-арної купи: {dary_d}")

        results: Dict[str, List[BenchmarkResult]] = {
            'Prim (Binary)': [],
            f'Prim ({dary_d}-ary)': [],
            'Prim (Indexed)': [],
            'Kruskal': [],
        }

        # обгортка щоб передати d у конструктор PrimMSTDAry
        def _dary_factory(g):
            return PrimMSTDAry(g, record_steps=False, d=dary_d)

        variants = [
            ('Prim (Binary)', PrimMST),
            (f'Prim ({dary_d}-ary)', _dary_factory),
            ('Prim (Indexed)', PrimMSTIndexed),
            ('Kruskal', KruskalMST),
        ]

        for n in sizes:
            gen = GraphGenerator(seed=42 + n)
            if graph_type == 'random':
                g = gen.random_graph(n, complete=True)
            elif graph_type == 'sparse':
                edge_target = max(n - 1, 7 * n)
                g = gen.sparse_graph(num_nodes=n, num_edges=edge_target)
            elif graph_type == 'grid':
                rows = max(1, int(n ** 0.5))
                cols = max(1, n // rows)
                g = gen.grid_graph(rows, cols)
            else:
                raise ValueError(f"Невідомий graph_type: {graph_type}")

            if verbose:
                density = _compute_density(g)
                print(f"\n  V={g.node_count()}, E={g.edge_count()} "
                      f"(щільність: {density:.1%})")

            for name, klass_or_factory in variants:
                if name == f'Prim ({dary_d}-ary)':
                    res = self._measure_with_factory(klass_or_factory, g,
                                                     name, verbose=verbose)
                else:
                    res = self.measure_full(klass_or_factory, g,
                                            verbose=verbose)
                results[name].append(res)

        return results

    def _measure_with_factory(self, factory, graph: Graph,
                              algo_name: str, verbose: bool = False,
                              memory_iterations: int = 10) -> BenchmarkResult:
        """measure_full() для алгоритмів з нестандартним конструктором
        (наприклад PrimMSTDAry приймає d). factory(graph) → готовий об'єкт."""
        for _ in range(self.config.warmup_iterations):
            factory(graph).find_mst()

        # -- Пробний запуск --
        start = time.perf_counter()
        reference = factory(graph).find_mst()
        single_run = time.perf_counter() - start
        iterations = self.config.calculate_iterations(single_run)

        if verbose:
            print(f"  {algo_name}: ~{single_run*1000:.2f} мс/запуск -> "
                  f"{iterations} ітерацій")

        # час
        raw_times = []
        gc_was = gc.isenabled()
        gc.disable()
        try:
            for _ in range(iterations):
                algo = factory(graph)
                t0 = time.perf_counter()
                algo.find_mst()
                raw_times.append(time.perf_counter() - t0)
        finally:
            if gc_was:
                gc.enable()

        filtered = _iqr_filter(raw_times)
        stats = _compute_stats(filtered)

        if stats['cv'] > self.config.cv_warning_threshold:
            warnings.warn(
                f"Нестабільні заміри для {algo_name} "
                f"(V={graph.node_count()}): CV={stats['cv']:.1%}",
                RuntimeWarning
            )

        # пам'ять — окрема серія з tracemalloc
        peaks = []
        for _ in range(memory_iterations):
            gc.collect()
            tracemalloc.start()
            try:
                factory(graph).find_mst()
                _, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
            peaks.append(peak)

        peak_mean_kb = statistics.mean(peaks) / 1024
        peak_std_kb = (statistics.stdev(peaks) / 1024) if len(peaks) > 1 else 0

        return BenchmarkResult(
            algorithm_name=algo_name,
            node_count=graph.node_count(),
            edge_count=graph.edge_count(),
            density=_compute_density(graph),
            iterations=iterations,
            iterations_after_filter=len(filtered),
            mean_time=stats['mean'],
            std_time=stats['std'],
            median_time=stats['median'],
            min_time=stats['min'],
            max_time=stats['max'],
            ci_lower=stats['ci_lower'],
            ci_upper=stats['ci_upper'],
            cv=stats['cv'],
            mst_cost=reference.total_cost,
            peak_memory_kb=peak_mean_kb,
            peak_memory_std_kb=peak_std_kb,
            memory_iterations=memory_iterations,
        )

    def compare_dary_d_values(self, graph: Graph,
                              d_values: List[int] = None,
                              verbose: bool = True
                              ) -> Dict[int, BenchmarkResult]:
        """Як арність d впливає на час Пріма з d-арною купою."""
        if d_values is None:
            d_values = [2, 3, 4, 5, 8, 16]

        if verbose:
            print(f"\n  Вплив параметра d | V={graph.node_count()}, E={graph.edge_count()}")
            print(f"  Тестовані d: {d_values}")

        results: Dict[int, BenchmarkResult] = {}
        for d in d_values:
            factory = lambda g, dd=d: PrimMSTDAry(g, record_steps=False, d=dd)
            res = self._measure_with_factory(
                factory, graph, f"Prim ({d}-ary)",
                verbose=verbose, memory_iterations=5
            )
            results[d] = res
        return results

    # ВПЛИВ ЩІЛЬНОСТІ

    def test_density_impact(self, node_count: int = 200,
                            verbose: bool = True) -> Dict[str, List[BenchmarkResult]]:
        """
        Тестує вплив щільності ребер на швидкість алгоритмів.

        Методологія: фіксується V, генерується набір вершин,
        потім для кожної щільності p створюється граф,
        де кожне ребро включається з ймовірністю p.
        Зв'язність гарантується backbone-ланцюгом 0-1-2-...(n-1).
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"  Вплив щільності | V={node_count}")
            print(f"{'='*60}")

        # Генеруємо вершини (без ребер)
        gen = GraphGenerator(seed=42)
        base = gen.random_graph(node_count, complete=False)
        nodes = list(base.iter_nodes())
        node_ids = [n.id for n in nodes]

        # Обчислюємо всі можливі ребра наперед
        all_possible_edges = []
        backbone_set = set()

        for i in range(len(node_ids)):
            if i < len(node_ids) - 1:
                a, b = min(node_ids[i], node_ids[i+1]), max(node_ids[i], node_ids[i+1])
                backbone_set.add((a, b))
            for j in range(i + 1, len(node_ids)):
                n1, n2 = nodes[i], nodes[j]
                dist = n1.distance_to(n2)
                a, b = min(n1.id, n2.id), max(n1.id, n2.id)
                all_possible_edges.append((a, b, dist))

        probabilities = [0.02, 0.05, 0.1, 0.2, 0.4, 0.7, 1.0]
        prim_results = []
        kruskal_results = []
        rng = random.Random(42)

        for p in probabilities:
            g = Graph()
            for node in nodes:
                g.add_node(Node(node.id, node.name, node.x, node.y))

            added = set()

            # Backbone (завжди)
            for a, b, dist in all_possible_edges:
                if (a, b) in backbone_set and (a, b) not in added:
                    g.add_edge(Edge(a, b, dist, 150.0))
                    added.add((a, b))

            # Решта ребер з ймовірністю p
            for a, b, dist in all_possible_edges:
                if (a, b) not in added and rng.random() <= p:
                    g.add_edge(Edge(a, b, dist, 150.0))
                    added.add((a, b))

            actual_density = _compute_density(g)
            if verbose:
                print(f"\n  p={p:.0%} -> E={g.edge_count()}, "
                      f"щільність={actual_density:.1%}")

            prim_results.append(self.measure_full(PrimMST, g, verbose=verbose))
            kruskal_results.append(self.measure_full(KruskalMST, g, verbose=verbose))

        return {'PrimMST': prim_results, 'KruskalMST': kruskal_results}

    # ДИНАМІЧНИЙ ТА СТАТИЧНИЙ

    def test_dynamic_vs_static(self, node_count: int = 500,
                               num_trials: int = 20,
                               verbose: bool = True) -> Dict[str, float]:
        """Static (повна перебудова Прімом) проти Dynamic (інкрементальне додавання)."""
        if verbose:
            print(f"\n  Dynamic vs Static | V={node_count}, trials={num_trials}")

        gen = GraphGenerator(seed=123)
        base_graph = gen.random_graph(node_count, complete=True)

        static_times = []
        dynamic_times = []

        gc_was_enabled = gc.isenabled()
        gc.disable()
        try:
            for trial in range(num_trials):
                new_node = Node(
                    10000 + trial,
                    f"Нова-{trial}",
                    48.0 + trial * 0.001,
                    35.0 + trial * 0.001
                )

                # -- Static: клон + повна перебудова --
                static_graph = Graph.from_dict(base_graph.to_dict())
                static_graph.add_node(Node(
                    new_node.id, new_node.name, new_node.x, new_node.y
                ))
                for nid in base_graph.get_node_ids():
                    other = base_graph.get_node(nid)
                    dist = new_node.distance_to(other)
                    static_graph.add_edge(
                        Edge(new_node.id, nid, dist, 150.0)
                    )

                start = time.perf_counter()
                PrimMST(static_graph, record_steps=False).find_mst()
                static_times.append(time.perf_counter() - start)

                # -- Dynamic: інкрементально --
                dyn = DynamicMST(Graph.from_dict(base_graph.to_dict()))

                start = time.perf_counter()
                dyn.add_node(Node(
                    new_node.id, new_node.name, new_node.x, new_node.y
                ), cost_per_meter=150.0)
                dynamic_times.append(time.perf_counter() - start)
        finally:
            if gc_was_enabled:
                gc.enable()

        # Фільтрація та статистика
        static_filtered = _iqr_filter(static_times)
        dynamic_filtered = _iqr_filter(dynamic_times)

        static_stats = _compute_stats(static_filtered)
        dynamic_stats = _compute_stats(dynamic_filtered)

        static_ms = static_stats['mean'] * 1000
        dynamic_ms = dynamic_stats['mean'] * 1000
        speedup = static_ms / dynamic_ms if dynamic_ms > 0 else float('inf')

        # -- Окремо: пам'ять для Static та Dynamic --
        static_peaks = []
        dynamic_peaks = []
        mem_iters = min(5, num_trials)
        for trial in range(mem_iters):
            new_node = Node(
                20000 + trial, f"Mem-{trial}",
                48.0 + trial * 0.001, 35.0 + trial * 0.001
            )

            # Static
            sg = Graph.from_dict(base_graph.to_dict())
            sg.add_node(Node(new_node.id, new_node.name, new_node.x, new_node.y))
            for nid in base_graph.get_node_ids():
                other = base_graph.get_node(nid)
                sg.add_edge(Edge(new_node.id, nid,
                                 new_node.distance_to(other), 150.0))
            gc.collect()
            tracemalloc.start()
            try:
                PrimMST(sg, record_steps=False).find_mst()
                _, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
            static_peaks.append(peak)

            # Dynamic
            dyn = DynamicMST(Graph.from_dict(base_graph.to_dict()))
            gc.collect()
            tracemalloc.start()
            try:
                dyn.add_node(Node(new_node.id, new_node.name,
                                  new_node.x, new_node.y), cost_per_meter=150.0)
                _, peak = tracemalloc.get_traced_memory()
            finally:
                tracemalloc.stop()
            dynamic_peaks.append(peak)

        static_mem_kb = statistics.mean(static_peaks) / 1024
        dynamic_mem_kb = statistics.mean(dynamic_peaks) / 1024

        if verbose:
            print(f"  Static:  {static_ms:.3f} +/- "
                  f"{static_stats['std']*1000:.3f} мс (n={len(static_filtered)}), "
                  f"пам'ять: {static_mem_kb:,.1f} КБ")
            print(f"  Dynamic: {dynamic_ms:.3f} +/- "
                  f"{dynamic_stats['std']*1000:.3f} мс (n={len(dynamic_filtered)}), "
                  f"пам'ять: {dynamic_mem_kb:,.1f} КБ")
            print(f"  Прискорення: {speedup:,.1f}x")

        return {
            'static_ms': static_ms,
            'dynamic_ms': dynamic_ms,
            'static_std_ms': static_stats['std'] * 1000,
            'dynamic_std_ms': dynamic_stats['std'] * 1000,
            'static_memory_kb': static_mem_kb,
            'dynamic_memory_kb': dynamic_mem_kb,
            'speedup': speedup,
            'node_count': node_count,
        }


    # ФОРМАТОВАНИЙ ЗВІТ

    @staticmethod
    def format_results_table(results: Dict[str, List[BenchmarkResult]]) -> str:
        """
        Генерує текстову таблицю результатів.

        Returns:
            Відформатована таблиця з усіма статистиками.
        """
        lines = []
        header = (
            f"{'V':>5} {'E':>7} {'Щільн.':>7} "
            f"{'Алгоритм':<25} "
            f"{'Mean+/-Std (мс)':>16} {'95% CI (мс)':>20} "
            f"{'Медіана':>9} {'CV':>6} {'n':>4} {'Пам.КБ':>9}"
        )
        lines.append(header)
        lines.append("-" * len(header))

        prim_res = results.get('PrimMST', [])
        kruskal_res = results.get('KruskalMST', [])

        for p, k in zip(prim_res, kruskal_res):
            for r in [p, k]:
                mem_col = f"{r.peak_memory_kb:>9.1f}" if r.peak_memory_kb > 0 else f"{'—':>9}"
                lines.append(
                    f"{r.node_count:>5} {r.edge_count:>7} {r.density:>6.1%} "
                    f"{r.algorithm_name:<25} "
                    f"{r.mean_time_ms:>7.3f}+/-{r.std_time_ms:<7.3f} "
                    f"[{r.ci_lower_ms:>7.3f}, {r.ci_upper_ms:>7.3f}] "
                    f"{r.median_time*1000:>8.3f} {r.cv:>5.1%} "
                    f"{r.iterations_after_filter:>4} {mem_col}"
                )
            lines.append("")

        return "\n".join(lines)


# ЗАПУСК З КОМАНДНОГО РЯДКА

if __name__ == "__main__":
    bench = Benchmark()

    print("=" * 60)
    print("  БЕНЧМАРКИ ДЛЯ КУРСОВОЇ РОБОТИ")
    print("=" * 60)

    # 1. Порівняння алгоритмів
    results = bench.full_comparison(
        sizes=[50, 100, 200, 500, 1000],
        graph_type='random',
        verbose=True
    )
    print("\n" + bench.format_results_table(results))

    # 2. Вплив щільності
    density_results = bench.test_density_impact(
        node_count=200,
        verbose=True
    )

    # 3. Dynamic vs Static
    dynamic_results = bench.test_dynamic_vs_static(
        node_count=300,
        num_trials=20,
        verbose=True
    )