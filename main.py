"""
main.py — Головний модуль системи оптимізації енергомережі

Запуск:
    python main.py              # Повна демонстрація
    python main.py --benchmark  # Тільки бенчмарки (генерує PDF графіки)
    python main.py --visual     # Тільки візуалізація
    python main.py --dynamic    # Тільки динамічне МКД
    python main.py --generate   # Демонстрація генерації графів
    python main.py --io         # Демонстрація завантаження/збереження
    python main.py --db         # Демонстрація роботи з базою даних (SQLite)
    python main.py --load FILE  # Завантажити граф з файлу
"""

import argparse
import os
from typing import List, Tuple

from models import Node, Edge, Graph
from algorithms import (
    PrimMST, PrimMSTDAry, PrimMSTIndexed, KruskalMST, DynamicMST,
    augment_to_2_edge_connected, find_bridges, is_2_edge_connected,
)
from visualization import StaticVisualizer, StepByStepVisualizer
from analysis import Benchmark, ComplexityAnalyzer
from data import GraphGenerator, DataLoader, DataExporter, DatabaseManager
from data.demo import create_demo_graph, DEMO_NODES, COST_PER_METER


def print_section(title: str) -> None:
    """Друкує заголовок секції."""
    print(f"\n{'─' * 70}")
    print(f"🔹 {title}")
    print(f"{'─' * 70}")


# ДЕМОНСТРАЦІЇ

def demo_algorithms():
    """Демонстрація алгоритмів Пріма та Крускала на 25 ПС."""
    print("\n" + "=" * 70)
    print("   ДЕМОНСТРАЦІЯ АЛГОРИТМІВ ПОБУДОВИ МКД")
    print("   Мережа: 25 підстанцій")
    print("=" * 70)

    graph = create_demo_graph()

    print(f"\n Вхідні дані:")
    print(f"   Підстанцій: {graph.node_count()}")
    print(f"   Можливих ліній: {graph.edge_count()}")
    print(f"   Вартість прокладання: {COST_PER_METER} грн/м")

    # Алгоритм Пріма
    print_section("АЛГОРИТМ ПРІМА (бінарна купа)")

    prim = PrimMST(graph, record_steps=True)
    result_prim = prim.find_mst(track_memory=True)

    print(f"   Час виконання: {result_prim.execution_time * 1000:.3f} мс")
    print(f"   Пам'ять (пік): {result_prim.peak_memory_kb:,.1f} КБ")
    print(f"   Ребер у МКД: {len(result_prim.edges)}")
    print(f"   Загальна довжина: {result_prim.total_distance / 1000:,.1f} км")
    print(f"   Загальна вартість: {result_prim.total_cost:,.2f} грн")

    print("\n   Ребра МКД:")
    for i, edge in enumerate(result_prim.edges, 1):
        n1 = graph.get_node(edge.node1)
        n2 = graph.get_node(edge.node2)
        print(f"   {i:2d}. {n1.name} — {n2.name}: {edge.distance / 1000:.1f} км, {edge.weight:,.0f} грн")

    # Алгоритм Крускала
    print_section("АЛГОРИТМ КРУСКАЛА (Union-Find)")

    kruskal = KruskalMST(graph, record_steps=True)
    result_kruskal = kruskal.find_mst(track_memory=True)

    print(f"   Час виконання: {result_kruskal.execution_time * 1000:.3f} мс")
    print(f"   Пам'ять (пік): {result_kruskal.peak_memory_kb:,.1f} КБ")
    print(f"   Ребер у МКД: {len(result_kruskal.edges)}")
    print(f"   Загальна довжина: {result_kruskal.total_distance / 1000:,.1f} км")
    print(f"   Загальна вартість: {result_kruskal.total_cost:,.2f} грн")

    # Перехресна верифікація
    diff = abs(result_prim.total_cost - result_kruskal.total_cost)
    if diff < 0.01:
        print(f"\n✅ Перехресна верифікація: вартість збігається ({result_prim.total_cost:,.2f} грн)")
    else:
        print(f"\n⚠️ Результати відрізняються на {diff:,.2f} грн!")

    return graph, result_prim


def demo_visualization(graph: Graph, result):
    """Генерація статичних зображень та GIF-анімації."""
    print("\n" + "=" * 70)
    print("   ВІЗУАЛІЗАЦІЯ РЕЗУЛЬТАТІВ (matplotlib)")
    print("=" * 70)

    # Статичний графік
    print("\n Створюю статичний графік...")
    viz = StaticVisualizer(graph, result)
    viz.save("output/mst_static.png")
    print("   Збережено: output/mst_static.png")

    # Покрокова анімація
    print("\n Створюю покрокову анімацію...")
    animator = StepByStepVisualizer(graph, result)
    animator.save_gif("output/mst_animation.gif", interval=1200)
    print("   Збережено: output/mst_animation.gif")


def demo_dynamic():
    """Демонстрація динамічного оновлення МКД."""
    print("\n" + "=" * 70)
    print("   ДИНАМІЧНЕ ОНОВЛЕННЯ МКД")
    print("=" * 70)

    # Починаємо з 20 підстанцій, потім додаємо по одній
    graph = Graph()
    for node in DEMO_NODES[:20]:
        graph.add_node(node)
    graph.generate_complete_graph(cost_per_meter=COST_PER_METER)

    dynamic = DynamicMST(graph)

    print(f"\nПочатковий стан: {graph.node_count()} підстанцій")
    print(f"   Вартість МКД: {dynamic.total_cost:,.2f} грн")

    # Послідовно додаємо 5 підстанцій
    for node in DEMO_NODES[20:]:
        print_section(f"Введення в експлуатацію: {node.name}")

        result = dynamic.add_node(node, cost_per_meter=COST_PER_METER)
        method = "інкрементальний" if not result.rebuild_required else "перебудова"
        print(f"   Метод: {method}")
        print(f"   Час: {result.execution_time * 1000:.3f} мс")
        print(f"   Підстанцій: {dynamic.graph.node_count()}")
        print(f"   Вартість МКД: {result.total_cost:,.2f} грн")

    # Порівняння з повною перебудовою
    full_graph = create_demo_graph()
    full_result = PrimMST(full_graph).find_mst()
    diff = abs(dynamic.total_cost - full_result.total_cost)
    if diff < 0.01:
        print(f"\nВерифікація: інкрементальний результат = повна перебудова ({full_result.total_cost:,.2f} грн)")
    else:
        print(f"\n⚠️ Різниця: {diff:,.2f} грн (можливий ефект нерівності трикутника)")

    # Візуалізація
    print(f"\n📊 Зберігаю результат...")
    mst_result = dynamic.get_mst_result()
    viz = StaticVisualizer(dynamic.graph, mst_result)
    viz.save("output/mst_dynamic.png")
    print("   Збережено: output/mst_dynamic.png")


def demo_2ec():
    """Демонстрація доповнення МКД до 2-edge-connected."""
    print("\n" + "=" * 70)
    print("   Резервування мережі (2-edge-connected augmentation)")
    print("=" * 70)

    graph = create_demo_graph()
    mst = PrimMST(graph).find_mst()

    print_section("Демо-мережа (25 ПС Дніпропетровської області)")
    print(f"   МКД: {len(mst.edges)} ребер, вартість {mst.total_cost:,.0f} грн")

    # для дерева кожне ребро — міст
    mst_graph = Graph()
    for n in graph.iter_nodes():
        mst_graph.add_node(Node(n.id, n.name, n.x, n.y))
    for e in mst.edges:
        mst_graph.add_edge(Edge(e.node1, e.node2, e.distance, e.cost_per_meter))

    bridges_before = find_bridges(mst_graph)
    print(f"   Мостів у МКД: {len(bridges_before)}")

    result = augment_to_2_edge_connected(graph, mst.edges, track_memory=True)
    print()
    print(result)

    print(f"\n   Резервні ребра:")
    for i, e in enumerate(result.reserve_edges, 1):
        n1 = graph.get_node(e.node1).name
        n2 = graph.get_node(e.node2).name
        print(f"   {i}. {n1} <-> {n2}: {e.distance/1000:.1f} км, "
              f"{e.weight:,.0f} грн")

    pct = result.reserve_cost / mst.total_cost * 100
    print(f"\n   Підсумок:")
    print(f"     Вартість МКД:    {mst.total_cost:,.0f} грн")
    print(f"     Вартість резерву: {result.reserve_cost:,.0f} грн (+{pct:.1f}%)")
    print(f"     Загалом:         {mst.total_cost + result.reserve_cost:,.0f} грн")

    try:
        from visualization import StaticVisualizer
        viz = StaticVisualizer(graph, mst, reserve_edges=result.reserve_edges)
        viz.plot(show_all_edges=False, show_mst=True,
                 title='МКД + Резервування (2-edge-connected)')
        viz.save("output/mst_2ec.png")
        print(f"\n   Збережено: output/mst_2ec.png")
    except Exception as e:
        print(f"\n   Помилка візуалізації: {e}")

    print_section("Масштабованість 2-EC-доповнення")
    print(f"   {'V':>5} {'Листків':>9} {'Резерв':>8} {'Опт.':>6} {'Ratio':>7} "
          f"{'Час (мс)':>10} {'Пам (КБ)':>10} {'+% варт.':>10}")
    print(f"   {'-'*78}")
    for V in [50, 100, 200, 500, 1000]:
        g = GraphGenerator(seed=42).random_graph(V, complete=True)
        m = PrimMST(g, record_steps=False).find_mst()
        r = augment_to_2_edge_connected(g, m.edges, track_memory=True)
        pct = r.reserve_cost / m.total_cost * 100
        print(f"   {V:>5} {r.leaves_count:>9} {r.edges_added:>8} "
              f"{r.lower_bound:>6} {r.approximation_ratio:>6.2f}x "
              f"{r.execution_time*1000:>10.1f} {r.peak_memory_kb:>10,.0f} "
              f"{pct:>9.1f}%")


def demo_benchmark():
    """ аналіз продуктивності з генерацією PDF-графіків."""
    print("\n" + "=" * 70)
    print(" БЕНЧМАРКИ")
    print("=" * 70)

    benchmark = Benchmark()

    # 1. Масштабованість
    print_section("Аналіз масштабованості (V=50...1000)")
    res_comp = benchmark.full_comparison(sizes=[50, 100, 200, 500, 1000], graph_type='random', verbose=True)

    # 1b. Референс-тест рецензента: розріджений 500 ПС, 4000 ребер
    print_section("Референс-тест: розріджений граф V=500, E=4000 (рецензент: 45/70 мс)")
    gen = GraphGenerator(seed=42)
    sparse_500 = gen.sparse_graph(num_nodes=500, num_edges=4000)
    print(f"   Згенеровано: V={sparse_500.node_count()}, E={sparse_500.edge_count()}, "
          f"щільність={2*sparse_500.edge_count()/(sparse_500.node_count()*(sparse_500.node_count()-1)):.2%}")
    p_ref = benchmark.measure_full(PrimMST, sparse_500, verbose=False)
    k_ref = benchmark.measure_full(KruskalMST, sparse_500, verbose=False)
    print(f"   Прім:    {p_ref.mean_time_ms:.2f} ± {p_ref.std_time_ms:.2f} мс, "
          f"пам'ять: {p_ref.peak_memory_kb:,.1f} КБ")
    print(f"   Крускал: {k_ref.mean_time_ms:.2f} ± {k_ref.std_time_ms:.2f} мс, "
          f"пам'ять: {k_ref.peak_memory_kb:,.1f} КБ")

    # 2. Вплив щільності
    print_section("Аналіз впливу щільності графа (V=200)")
    res_dens = benchmark.test_density_impact(node_count=200, verbose=True)

    # 3. Динамічне оновлення
    print_section("Ефективність динамічного оновлення (V=500)")
    res_dyn = benchmark.test_dynamic_vs_static(node_count=500, verbose=True)

    # 4. Порівняння варіантів Пріма (binary / d-ary / indexed) + Крускал
    print_section("Порівняння варіантів Пріма (Binary / 4-ary / Indexed) vs Крускал")
    res_variants = benchmark.compare_prim_variants(
        sizes=[100, 200, 500], graph_type='random', dary_d=4, verbose=True
    )

    # 5. Як параметр d впливає на час Пріма (для одного V)
    print_section("Вплив параметра d на час Пріма з d-арною купою (V=500)")
    sample_graph = GraphGenerator(seed=42).random_graph(500, complete=True)
    res_dary = benchmark.compare_dary_d_values(
        sample_graph, d_values=[2, 3, 4, 5, 8, 16], verbose=True
    )

    # 6. Масштабованість 2-EC доповнення
    print_section("Масштабованість 2-edge-connected доповнення")
    aug_results = []
    for V in [50, 100, 200, 500]:
        g = GraphGenerator(seed=42).random_graph(V, complete=True)
        m = PrimMST(g, record_steps=False).find_mst()
        r = augment_to_2_edge_connected(g, m.edges, track_memory=True)
        pct = r.reserve_cost / m.total_cost * 100
        aug_results.append({
            'node_count': V,
            'edges_added': r.edges_added,
            'lower_bound': r.lower_bound,
            'execution_time_ms': r.execution_time * 1000,
            'peak_memory_kb': r.peak_memory_kb,
            'reserve_cost_pct': pct,
        })
        print(f"   V={V}: +{r.edges_added} ребер ({r.approximation_ratio:.2f}x опт.), "
              f"{r.execution_time*1000:.1f} мс, {r.peak_memory_kb:.0f} КБ, +{pct:.1f}% вартості")

    # PDF-графіки
    print("\nСтворюю PDF-графіки...")
    analyzer = ComplexityAnalyzer()
    saved = analyzer.save_all_plots(
        results=res_comp,
        density_results=res_dens,
        dynamic_res=res_dyn,
        prim_variants_res=res_variants,
        dary_d_res=res_dary,
        augmentation_res=aug_results,
        output_dir="output_plots"
    )
    print(f"   Збережено {len(saved)} графіків у папку output_plots/")


def demo_generation():
    """Демонстрація генерації тестових графів."""
    print("\n" + "=" * 70)
    print("   ГЕНЕРАЦІЯ ТЕСТОВИХ ГРАФІВ")
    print("=" * 70)

    generator = GraphGenerator(seed=42)

    configs = [
        ("ВИПАДКОВИЙ ГРАФ (20 ПС)",
         lambda: generator.random_graph(num_nodes=20, cost_per_meter=COST_PER_METER)),
        ("СІТКОВИЙ ГРАФ (4×4)",
         lambda: generator.grid_graph(rows=4, cols=4, cost_per_meter=COST_PER_METER)),
        ("КЛАСТЕРНИЙ ГРАФ (3 кластери × 5 ПС)",
         lambda: generator.cluster_graph(num_clusters=3, nodes_per_cluster=5, cost_per_meter=COST_PER_METER)),
    ]

    for title, gen_func in configs:
        print_section(title)
        graph = gen_func()
        result = PrimMST(graph).find_mst()
        print(f"   Вершин: {graph.node_count()}, Ребер: {graph.edge_count()}")
        print(f"   МКД: {len(result.edges)} ребер, вартість: {result.total_cost:,.0f} грн")


def demo_io():
    """Демонстрація експорту результатів."""
    print("\n" + "=" * 70)
    print("   ЕКСПОРТ РЕЗУЛЬТАТІВ")
    print("=" * 70)

    graph = create_demo_graph()
    result = PrimMST(graph).find_mst()

    print_section("ЕКСПОРТ")
    DataExporter.to_json(result, graph, "output/mst_result.json")
    print("   JSON: output/mst_result.json")
    DataExporter.to_csv(result, graph, "output/mst_edges.csv")
    print("   CSV:  output/mst_edges.csv")


def demo_database():
    """Демонстрація збереження та завантаження з SQLite."""
    print("\n" + "=" * 70)
    print("   БАЗА ДАНИХ SQLITE")
    print("=" * 70)

    db = DatabaseManager("energy_network.db")
    graph = create_demo_graph()

    # Зберігаємо мережу
    net_id = db.save_network("Дніпропетровська область (25 ПС)", graph, cost_per_meter=COST_PER_METER)
    print(f"   Мережу збережено. ID = {net_id}")

    # Два алгоритми → історія
    result_prim = PrimMST(graph).find_mst()
    db.save_mst_result(net_id, result_prim, "Prim (Binary Heap)")

    result_kruskal = KruskalMST(graph).find_mst()
    db.save_mst_result(net_id, result_kruskal, "Kruskal (Union-Find)")

    # Історія
    history = db.get_mst_results(net_id)
    print(f"\n   Історія запусків для мережі ID {net_id}:")
    for r in history:
        print(f"      {r['algorithm']}: {r['total_cost']:,.0f} грн, {r.get('execution_time', 0) * 1000:.2f} мс")

    # Завантаження назад
    loaded = db.load_network(net_id)
    print(f"\n   ✅ Завантажено з БД: {loaded.node_count()} вершин, {loaded.edge_count()} ребер")


def run_with_file(filepath: str):
    """Завантаження графа з файлу та побудова МКД."""
    print(f"\n📂 Завантаження: {filepath}")
    try:
        graph = DataLoader.load(filepath)
        result = PrimMST(graph).find_mst()
        print(f"   Підстанцій: {graph.node_count()}")
        print(f"   Час: {result.execution_time * 1000:.3f} мс")
        print(f"   Вартість МКД: {result.total_cost:,.2f} грн")
    except Exception as e:
        print(f"   ❌ Помилка: {e}")

# ТОЧКА ВХОДУ


def main():
    parser = argparse.ArgumentParser(
        description="Оптимізація енергомережі за допомогою МКД",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--benchmark', action='store_true', help='Аналіз продуктивності (PDF-графіки)')
    parser.add_argument('--visual', action='store_true', help='Візуалізація (PNG + GIF)')
    parser.add_argument('--dynamic', action='store_true', help='Динамічне оновлення МКД')
    parser.add_argument('--generate', action='store_true', help='Генерація тестових графів')
    parser.add_argument('--io', action='store_true', help='Експорт результатів')
    parser.add_argument('--db', action='store_true', help='Робота з базою даних')
    parser.add_argument('--augment', action='store_true', help='Резервування мережі (2-edge-connected)')
    parser.add_argument('--all', action='store_true', help='Запустити все')
    parser.add_argument('--load', type=str, metavar='FILE', help='Завантажити граф з файлу')

    args = parser.parse_args()

    if args.load:
        run_with_file(args.load)
        return

    run_all = args.all or not any([args.benchmark, args.visual, args.dynamic,
                                    args.generate, args.io, args.db, args.augment])

    os.makedirs("output", exist_ok=True)
    os.makedirs("output_plots", exist_ok=True)

    print("\n" + "═" * 70)
    print("   СИСТЕМА ОПТИМІЗАЦІЇ КОНФІГУРАЦІЇ ЕНЕРГОМЕРЕЖІ")
    print("   Демонстраційна мережа: 25 ПС Дніпропетровської області")
    print("═" * 70)

    graph, result = None, None

    if run_all or args.visual:
        graph, result = demo_algorithms()
        demo_visualization(graph, result)
    if run_all or args.dynamic:
        demo_dynamic()
    if run_all or args.generate:
        demo_generation()
    if run_all or args.io:
        demo_io()
    if run_all or args.db:
        demo_database()
    if run_all or args.augment:
        demo_2ec()
    if run_all or args.benchmark:
        demo_benchmark()

    print("\n" + "═" * 70)
    print("   ЗАВЕРШЕНО")
    print(f"   Результати: output/ та output_plots/")
    print("═" * 70 + "\n")


if __name__ == "__main__":
    main()