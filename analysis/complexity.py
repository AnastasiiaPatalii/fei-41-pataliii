"""
PDF-графіки за результатами бенчмарків.

Error bars — 95% довірчий інтервал (CI), не std.
Теоретичні криві нормалізовані за першою точкою.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Optional, Tuple
import os

from .benchmarks import BenchmarkResult


class ComplexityAnalyzer:
    """
    Аналізатор та візуалізатор обчислювальної складності.
    Включає як базові метрики, так і глибокий математичний аналіз асимптотики.
    """

    # Кольорова палітра
    COLORS = {
        'prim': '#1F77B4',       # Глибокий синій
        'kruskal': '#FF7F0E',    # Яскравий помаранчевий
        'dynamic': '#2CA02C',    # Зелений
        'theoretical': '#7F7F7F',# Сірий
        'grid': '#E8E8E8',       # Світло-сірий
        'text': '#2C3E50',       # Темний
        # Варіанти Пріма (для plot_prim_variants)
        'prim_binary': '#1F77B4',   # Синій
        'prim_dary': '#9467BD',     # Фіолетовий
        'prim_indexed': '#17BECF',  # Бірюзовий
    }

    def __init__(self, figsize: Tuple[int, int] = (10, 6), dpi: int = 150):
        self.figsize = figsize
        self.dpi = dpi

        # Налаштування стилю
        plt.rcParams['font.size'] = 11
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['legend.fontsize'] = 10

    @staticmethod
    def _ci_errors(results: List[BenchmarkResult]) -> List[List[float]]:
        """Асиметричні error bars для matplotlib: [lower_diffs, upper_diffs]."""
        lower = [r.mean_time_ms - r.ci_lower_ms for r in results]
        upper = [r.ci_upper_ms - r.mean_time_ms for r in results]
        return [lower, upper]


    def plot_scalability(self,
                         results: List[BenchmarkResult],
                         title: Optional[str] = None,
                         show_theoretical: bool = True) -> plt.Figure:
        """Будує графік масштабованості для одного алгоритму."""
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        nodes = [r.node_count for r in results]
        times = [r.mean_time_ms for r in results]
        ci_err = self._ci_errors(results)

        algo_name = results[0].algorithm_name if results else "Algorithm"
        color = self.COLORS['prim'] if 'Prim' in algo_name else self.COLORS['kruskal']

        ax.errorbar(nodes, times, yerr=ci_err, fmt='o-', color=color, linewidth=2,
                    markersize=8, capsize=5, capthick=2, label=f'{algo_name} (виміряно)')

        if show_theoretical and len(nodes) > 1:
            n = np.array(nodes)
            theoretical = n ** 2 * np.log2(n)
            theoretical = theoretical / theoretical[0] * times[0]
            ax.plot(n, theoretical, '--', color=self.COLORS['theoretical'], linewidth=2,
                    alpha=0.7, label=r'Теоретично $O(V^2 \log V)$')

        ax.set_xlabel('Кількість вершин (V)', fontweight='medium')
        ax.set_ylabel('Час виконання (мс)', fontweight='medium')
        ax.set_title(title or f'Часова складність: {algo_name}', fontweight='bold')
        ax.legend(loc='upper left', framealpha=0.9)
        ax.grid(True, alpha=0.3, color=self.COLORS['grid'])
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

        plt.tight_layout()
        return fig

    def plot_comparison(self,
                        prim_results: List[BenchmarkResult],
                        kruskal_results: List[BenchmarkResult],
                        title: Optional[str] = None) -> plt.Figure:
        """Порівняльний графік двох алгоритмів з 95% CI."""
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        nodes_p = [r.node_count for r in prim_results]
        times_p = [r.mean_time_ms for r in prim_results]
        ci_p = self._ci_errors(prim_results)

        nodes_k = [r.node_count for r in kruskal_results]
        times_k = [r.mean_time_ms for r in kruskal_results]
        ci_k = self._ci_errors(kruskal_results)

        ax.errorbar(nodes_p, times_p, yerr=ci_p, fmt='o-', color=self.COLORS['prim'],
                    linewidth=2, markersize=8, capsize=5, label='Пріма (Binary Heap)')
        ax.errorbar(nodes_k, times_k, yerr=ci_k, fmt='s-', color=self.COLORS['kruskal'],
                    linewidth=2, markersize=8, capsize=5, label='Крускала (Union-Find)')

        ax.set_xlabel('Кількість вершин (V)', fontweight='medium')
        ax.set_ylabel('Час виконання (мс)', fontweight='medium')
        ax.set_title(title or 'Порівняння алгоритмів Пріма та Крускала', fontweight='bold')
        ax.legend(loc='upper left', framealpha=0.9)
        ax.grid(True, alpha=0.3, color=self.COLORS['grid'])
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

        # Анотація про CI
        ax.annotate('Планки похибок: 95% довірчий інтервал',
                     xy=(0.02, 0.98), xycoords='axes fraction',
                     fontsize=8, color='gray', va='top')

        plt.tight_layout()
        return fig

    def plot_speedup(self,
                     prim_results: List[BenchmarkResult],
                     kruskal_results: List[BenchmarkResult]) -> plt.Figure:
        """Графік відносного прискорення (стовпчики)."""
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        nodes = [r.node_count for r in prim_results]
        speedup = [k.mean_time / p.mean_time if p.mean_time > 0 else 1
                    for p, k in zip(prim_results, kruskal_results)]

        bars = ax.bar([str(n) for n in nodes], speedup,
                      color=[self.COLORS['prim'] if s < 1 else self.COLORS['kruskal'] for s in speedup],
                      alpha=0.8, edgecolor='white', linewidth=2)

        ax.axhline(y=1, color='gray', linestyle='--', linewidth=2, alpha=0.5)

        for bar, s in zip(bars, speedup):
            height = bar.get_height()
            ax.annotate(f'{s:.2f}x', xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 5), textcoords="offset points", ha='center',
                        fontsize=10, fontweight='bold')

        ax.set_xlabel('Кількість вершин (V)', fontweight='medium')
        ax.set_ylabel('Відношення часу (Крускала / Пріма)', fontweight='medium')
        ax.set_title('Відносна продуктивність алгоритмів', fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        return fig

    def plot_edges_vs_time(self,
                           results: List[BenchmarkResult],
                           title: Optional[str] = None) -> plt.Figure:
        """Графік залежності часу від кількості ребер."""
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        edges = [r.edge_count for r in results]
        times = [r.mean_time_ms for r in results]
        ci_err = self._ci_errors(results)

        algo_name = results[0].algorithm_name if results else "Algorithm"
        color = self.COLORS['prim'] if 'Prim' in algo_name else self.COLORS['kruskal']

        ax.errorbar(edges, times, yerr=ci_err, fmt='o-', color=color, linewidth=2,
                    markersize=8, capsize=5, label=f'{algo_name}')

        if len(edges) > 1 and edges[0] > 0:
            e = np.array(edges)
            theoretical = e * np.log2(e)
            theoretical = theoretical / theoretical[0] * times[0]
            ax.plot(e, theoretical, '--', color=self.COLORS['theoretical'], linewidth=2,
                    alpha=0.7, label=r'Теоретично $O(E \log E)$')

        ax.set_xlabel('Кількість ребер (E)', fontweight='medium')
        ax.set_ylabel('Час виконання (мс)', fontweight='medium')
        ax.set_title(title or 'Залежність часу від кількості ребер', fontweight='bold')
        ax.legend(loc='upper left', framealpha=0.9)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def create_summary_figure(self,
                              prim_results: List[BenchmarkResult],
                              kruskal_results: List[BenchmarkResult]) -> plt.Figure:
        """Створює зведену фігуру 2x2."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=self.dpi)
        nodes = [r.node_count for r in prim_results]

        # 1. Порівняння часу (верхній лівий)
        ax1 = axes[0, 0]
        ci_p = self._ci_errors(prim_results)
        ci_k = self._ci_errors(kruskal_results)
        ax1.errorbar(nodes, [r.mean_time_ms for r in prim_results],
                     yerr=ci_p, fmt='o-', color=self.COLORS['prim'], label='Пріма', capsize=3)
        ax1.errorbar(nodes, [r.mean_time_ms for r in kruskal_results],
                     yerr=ci_k, fmt='s-', color=self.COLORS['kruskal'], label='Крускала', capsize=3)
        ax1.set_xlabel('Кількість вершин (V)')
        ax1.set_ylabel('Час (мс)')
        ax1.set_title('Порівняння часу виконання')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Логарифмічна шкала (верхній правий)
        ax2 = axes[0, 1]
        ax2.plot(nodes, [r.mean_time_ms for r in prim_results],
                 'o-', color=self.COLORS['prim'], label='Пріма')
        ax2.plot(nodes, [r.mean_time_ms for r in kruskal_results],
                 's-', color=self.COLORS['kruskal'], label='Крускала')
        ax2.set_xlabel('Кількість вершин (V)')
        ax2.set_ylabel('Час (мс, log шкала)')
        ax2.set_title('Логарифмічна шкала')
        ax2.set_yscale('log')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. Відношення часу (нижній лівий)
        ax3 = axes[1, 0]
        speedup = [k.mean_time / p.mean_time if p.mean_time > 0 else 1
                    for p, k in zip(prim_results, kruskal_results)]
        colors = [self.COLORS['prim'] if s > 1 else self.COLORS['kruskal'] for s in speedup]
        ax3.bar([str(n) for n in nodes], speedup, color=colors, alpha=0.8, edgecolor='white')
        ax3.axhline(y=1, color='gray', linestyle='--', linewidth=2)
        ax3.set_xlabel('Кількість вершин (V)')
        ax3.set_ylabel('Крускала / Пріма')
        ax3.set_title('Відносна продуктивність')
        ax3.grid(True, alpha=0.3, axis='y')

        # 4. Залежність від E (нижній правий)
        ax4 = axes[1, 1]
        edges_p = [r.edge_count for r in prim_results]
        ax4.plot(edges_p, [r.mean_time_ms for r in prim_results],
                 'o-', color=self.COLORS['prim'], label='Пріма')
        ax4.plot(edges_p, [r.mean_time_ms for r in kruskal_results],
                 's-', color=self.COLORS['kruskal'], label='Крускала')
        ax4.set_xlabel('Кількість ребер (E)')
        ax4.set_ylabel('Час (мс)')
        ax4.set_title('Залежність від кількості ребер')
        ax4.legend()
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_log_log_complexity(self,
                                prim_results: List[BenchmarkResult],
                                kruskal_results: List[BenchmarkResult]) -> plt.Figure:
        """Доказ поліноміальної складності (Log-Log Plot)."""
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        nodes = [r.node_count for r in prim_results]
        times_p = [r.mean_time_ms for r in prim_results]
        times_k = [r.mean_time_ms for r in kruskal_results]

        ax.plot(nodes, times_p, 'o-', color=self.COLORS['prim'], linewidth=2, label='Пріма')
        ax.plot(nodes, times_k, 's-', color=self.COLORS['kruskal'], linewidth=2, label='Крускала')

        n = np.array(nodes)
        theory_v2 = n**2
        theory_v2 = theory_v2 / theory_v2[-1] * times_k[-1]
        ax.plot(n, theory_v2, '--', color=self.COLORS['theoretical'], label=r'Нахил $\sim V^2$')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'Кількість вершин ($V$) [Log]')
        ax.set_ylabel('Час (мс) [Log]')
        ax.set_title("Доказ поліноміальної складності (Log-Log Plot)", fontweight='bold', pad=15)
        ax.legend(loc='upper left', framealpha=0.9)
        ax.grid(True, which="both", alpha=0.3, linestyle='--')

        plt.tight_layout()
        return fig

    def plot_density_impact(self,
                            prim_results: List[BenchmarkResult],
                            kruskal_results: List[BenchmarkResult]) -> plt.Figure:
        """Аналіз впливу щільності (Crossover point)."""
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        densities = [r.density * 100 for r in prim_results]
        ci_p = self._ci_errors(prim_results)
        ci_k = self._ci_errors(kruskal_results)

        ax.errorbar(densities, [r.mean_time_ms for r in prim_results],
                     yerr=ci_p, fmt='o-', color=self.COLORS['prim'],
                     linewidth=2.5, markersize=8, capsize=4, label='Пріма')
        ax.errorbar(densities, [r.mean_time_ms for r in kruskal_results],
                     yerr=ci_k, fmt='s-', color=self.COLORS['kruskal'],
                     linewidth=2.5, markersize=8, capsize=4, label='Крускала')

        v = prim_results[0].node_count
        ax.set_xlabel('Щільність графа (%)')
        ax.set_ylabel('Час виконання (мс)')
        ax.set_title(f"Вплив щільності ребер на продуктивність ($V={v}$)",
                     fontweight='bold', pad=15)
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.4, linestyle='--')
        ax.set_xlim(left=0, right=105)
        ax.set_ylim(bottom=0)

        # Знаходимо точку перетину
        for i in range(1, len(densities)):
            p_prev = prim_results[i-1].mean_time_ms
            p_curr = prim_results[i].mean_time_ms
            k_prev = kruskal_results[i-1].mean_time_ms
            k_curr = kruskal_results[i].mean_time_ms
            if (p_prev > k_prev and p_curr <= k_curr) or \
               (p_prev < k_prev and p_curr >= k_curr):
                ax.axvline(x=densities[i], color='red', linestyle=':', alpha=0.5)
                ax.text(densities[i] + 2, max(p_curr, k_curr) / 2,
                        'Crossover Point', color='red', rotation=90)
                break

        plt.tight_layout()
        return fig

    def plot_prim_variants(self,
                           results: Dict[str, List[BenchmarkResult]]) -> plt.Figure:
        """Binary / d-ary / Indexed Прім vs Крускал на одних графах. На вхід — вихід compare_prim_variants()."""
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        # Карта стилів для відомих ключів
        style_map = {
            'Binary': ('o-', self.COLORS['prim_binary']),
            'Indexed': ('D-', self.COLORS['prim_indexed']),
            'Kruskal': ('s-', self.COLORS['kruskal']),
        }
        default_style = ('^-', self.COLORS['prim_dary'])  # для d-арних

        for name, lst in results.items():
            if not lst:
                continue
            nodes = [r.node_count for r in lst]
            times = [r.mean_time_ms for r in lst]
            ci = self._ci_errors(lst)

            # Обираємо стиль за ключовим словом
            fmt, color = default_style
            for key, (f, c) in style_map.items():
                if key in name:
                    fmt, color = f, c
                    break

            ax.errorbar(nodes, times, yerr=ci, fmt=fmt, color=color,
                        linewidth=2, markersize=8, capsize=5, label=name)

        ax.set_xlabel('Кількість вершин (V)', fontweight='medium')
        ax.set_ylabel('Час виконання (мс)', fontweight='medium')
        ax.set_title('Порівняння варіантів реалізації Пріма',
                     fontweight='bold', pad=15)
        ax.legend(loc='upper left', framealpha=0.9)
        ax.grid(True, alpha=0.3, color=self.COLORS['grid'])
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.annotate('Планки похибок: 95% довірчий інтервал',
                    xy=(0.02, 0.02), xycoords='axes fraction',
                    fontsize=8, color='gray')

        plt.tight_layout()
        return fig

    def plot_dary_impact(self,
                         results_per_d: Dict[int, BenchmarkResult]) -> plt.Figure:
        """
        Як параметр d (арність купи) впливає на час Пріма.

        results_per_d: вихід Benchmark.compare_dary_d_values() — словник
        {d: BenchmarkResult} для одного графа з різними d.
        """
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        d_values = sorted(results_per_d.keys())
        times = [results_per_d[d].mean_time_ms for d in d_values]
        stds = [results_per_d[d].std_time_ms for d in d_values]

        ax.errorbar(d_values, times, yerr=stds, fmt='o-',
                    color=self.COLORS['prim_dary'],
                    linewidth=2, markersize=10, capsize=5,
                    label='Prim з d-арною купою')

        # Підписи значень над точками
        for d, t in zip(d_values, times):
            ax.annotate(f'{t:.1f}', xy=(d, t), xytext=(0, 8),
                        textcoords='offset points', ha='center',
                        fontsize=9, color=self.COLORS['text'])

        # Мітка про оптимальне d
        best_d = min(results_per_d, key=lambda d: results_per_d[d].mean_time)
        ax.axvline(x=best_d, color='red', linestyle=':', alpha=0.5)
        ax.text(best_d, max(times) * 0.95,
                f'оптимум: d={best_d}',
                color='red', ha='left' if best_d < max(d_values) / 2 else 'right',
                fontsize=10, fontweight='bold')

        sample = next(iter(results_per_d.values()))
        ax.set_xlabel('Арність купи (d)', fontweight='medium')
        ax.set_ylabel('Час виконання (мс)', fontweight='medium')
        ax.set_title(f'Вплив арності d на час Пріма '
                     f'(V={sample.node_count}, E={sample.edge_count})',
                     fontweight='bold', pad=15)
        ax.legend(loc='best', framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.set_xscale('log', base=2)
        ax.set_xticks(d_values)
        ax.set_xticklabels(d_values)

        plt.tight_layout()
        return fig

    def plot_augmentation_scalability(self,
                                      augmentation_results: List[Dict]) -> plt.Figure:
        """
        Графік масштабованості 2-edge-connected доповнення.

        Args:
            augmentation_results: список словників з полями:
                'node_count', 'edges_added', 'lower_bound',
                'execution_time_ms', 'peak_memory_kb', 'reserve_cost_pct'
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=self.dpi)

        nodes = [r['node_count'] for r in augmentation_results]
        times = [r['execution_time_ms'] for r in augmentation_results]
        memory = [r['peak_memory_kb'] for r in augmentation_results]
        added = [r['edges_added'] for r in augmentation_results]
        lower = [r['lower_bound'] for r in augmentation_results]

        # Лівий: час + пам'ять
        ax1 = axes[0]
        color_t = '#E74C3C'
        ax1.plot(nodes, times, 'o-', color=color_t, linewidth=2,
                 markersize=8, label='Час (мс)')
        ax1.set_xlabel('Кількість вершин (V)', fontweight='medium')
        ax1.set_ylabel('Час доповнення (мс)', color=color_t)
        ax1.tick_params(axis='y', labelcolor=color_t)
        ax1.grid(True, alpha=0.3)

        ax1m = ax1.twinx()
        color_m = self.COLORS['prim_dary']
        ax1m.plot(nodes, memory, 's--', color=color_m, linewidth=2,
                  markersize=8, alpha=0.7, label='Пам\'ять (КБ)')
        ax1m.set_ylabel('Пам\'ять (КБ)', color=color_m)
        ax1m.tick_params(axis='y', labelcolor=color_m)

        ax1.set_title('Час та пам\'ять 2-EC доповнення',
                      fontweight='bold', pad=10)

        # Правий: додано vs нижня межа
        ax2 = axes[1]
        x_idx = list(range(len(nodes)))
        width = 0.35
        ax2.bar([i - width/2 for i in x_idx], added, width,
                color=self.COLORS['kruskal'], label='Додано (наш алг.)',
                edgecolor='black', alpha=0.85)
        ax2.bar([i + width/2 for i in x_idx], lower, width,
                color=self.COLORS['dynamic'], label='Нижня межа ⌈L/2⌉',
                edgecolor='black', alpha=0.85)
        ax2.set_xticks(x_idx)
        ax2.set_xticklabels([str(v) for v in nodes])
        ax2.set_xlabel('Кількість вершин (V)', fontweight='medium')
        ax2.set_ylabel('Кількість резервних ребер')
        ax2.set_title('Точність апроксимації', fontweight='bold', pad=10)
        ax2.legend(loc='upper left')
        ax2.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        return fig

    def plot_memory_scalability(self,
                                prim_results: List[BenchmarkResult],
                                kruskal_results: List[BenchmarkResult]) -> plt.Figure:
        """Графік піку пам'яті залежно від кількості вершин."""
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)

        nodes_p = [r.node_count for r in prim_results]
        mem_p = [r.peak_memory_kb for r in prim_results]
        std_p = [r.peak_memory_std_kb for r in prim_results]

        nodes_k = [r.node_count for r in kruskal_results]
        mem_k = [r.peak_memory_kb for r in kruskal_results]
        std_k = [r.peak_memory_std_kb for r in kruskal_results]

        ax.errorbar(nodes_p, mem_p, yerr=std_p, fmt='o-', color=self.COLORS['prim'],
                    linewidth=2, markersize=8, capsize=5, label='Пріма (Binary Heap)')
        ax.errorbar(nodes_k, mem_k, yerr=std_k, fmt='s-', color=self.COLORS['kruskal'],
                    linewidth=2, markersize=8, capsize=5, label='Крускала (Union-Find)')

        # Теоретична O(V) для Пріма (heap зберігає до E елементів, але для повного
        # графа E = V²/2, тому пам'ять зростає квадратично разом із вхідним графом)
        if len(nodes_p) > 1 and mem_p[0] > 0:
            n = np.array(nodes_p)
            theory = n ** 2
            theory = theory / theory[0] * mem_p[0]
            ax.plot(n, theory, '--', color=self.COLORS['theoretical'], linewidth=2,
                    alpha=0.6, label=r'Теоретично $O(V^2)$ (для повного графа)')

        ax.set_xlabel('Кількість вершин (V)', fontweight='medium')
        ax.set_ylabel('Пік пам\'яті (КБ)', fontweight='medium')
        ax.set_title('Пам\'ять алгоритмів МКД (вимір через tracemalloc)',
                     fontweight='bold', pad=15)
        ax.legend(loc='upper left', framealpha=0.9)
        ax.grid(True, alpha=0.3, color=self.COLORS['grid'])
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

        ax.annotate('Планки похибок: ± станд. відхилення між ітераціями',
                    xy=(0.02, 0.98), xycoords='axes fraction',
                    fontsize=8, color='gray', va='top')

        plt.tight_layout()
        return fig

    def plot_dynamic_vs_static(self, static_ms: float, dynamic_ms: float,
                                v: int,
                                static_std_ms: float = 0,
                                dynamic_std_ms: float = 0) -> plt.Figure:
        """Порівняння повної перебудови та інкрементального оновлення."""
        fig, ax = plt.subplots(figsize=(8, 6), dpi=self.dpi)

        methods = ['Повна перебудова\n(Алгоритм Пріма)', 'Інкрементально\n(Dynamic MST)']
        times = [static_ms, dynamic_ms]
        errors = [static_std_ms, dynamic_std_ms]

        bars = ax.bar(methods, times,
                      yerr=errors if any(e > 0 for e in errors) else None,
                      capsize=8, color=[self.COLORS['prim'], self.COLORS['dynamic']],
                      alpha=0.8, edgecolor='black')

        ax.set_yscale('log')
        ax.set_ylabel('Час виконання (мс, log шкала)')
        ax.set_title(f"Ефективність динамічного МКД ($V={v}$)",
                     fontweight='bold', pad=15)

        for bar, t in zip(bars, times):
            ax.annotate(f'{t:.3f} мс',
                        xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 5), textcoords="offset points",
                        ha='center', va='bottom', fontweight='bold')

        speedup = static_ms / dynamic_ms if dynamic_ms > 0 else 0
        ax.text(0.5, 0.85, f"Прискорення: {speedup:,.0f}x",
                transform=ax.transAxes, ha='center', va='center',
                fontsize=14, fontweight='bold', color='red',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        ax.grid(True, axis='y', which='major', alpha=0.4)
        plt.tight_layout()
        return fig

    def save_all_plots(self,
                       results: Dict[str, List[BenchmarkResult]],
                       output_dir: str = "output_plots",
                       density_results: Dict[str, List[BenchmarkResult]] = None,
                       dynamic_res: Dict[str, float] = None,
                       prim_variants_res: Dict[str, List[BenchmarkResult]] = None,
                       dary_d_res: Dict[int, BenchmarkResult] = None,
                       augmentation_res: List[Dict] = None) -> List[str]:
        """Зберігає усі наявні графіки у output_dir у форматі PDF."""
        os.makedirs(output_dir, exist_ok=True)
        saved = []
        print(f"Збереження графіків у папку '{output_dir}'...")

        prim_res = results.get('PrimMST', [])
        kruskal_res = results.get('KruskalMST', [])

        if prim_res:
            fig = self.plot_scalability(prim_res, title="Масштабованість: Пріма")
            path = os.path.join(output_dir, '1_prim_scalability.pdf')
            fig.savefig(path, bbox_inches='tight')
            plt.close(fig)
            saved.append(path)

        if prim_res and kruskal_res:
            fig = self.plot_comparison(prim_res, kruskal_res)
            path = os.path.join(output_dir, '2_algorithm_comparison.pdf')
            fig.savefig(path, bbox_inches='tight')
            plt.close(fig)
            saved.append(path)

            fig = self.plot_speedup(prim_res, kruskal_res)
            path = os.path.join(output_dir, '3_speedup_bar.pdf')
            fig.savefig(path, bbox_inches='tight')
            plt.close(fig)
            saved.append(path)

            fig = self.plot_edges_vs_time(prim_res)
            path = os.path.join(output_dir, '4_edges_vs_time.pdf')
            fig.savefig(path, bbox_inches='tight')
            plt.close(fig)
            saved.append(path)

            fig = self.create_summary_figure(prim_res, kruskal_res)
            path = os.path.join(output_dir, '5_summary_grid.pdf')
            fig.savefig(path, bbox_inches='tight')
            plt.close(fig)
            saved.append(path)

            fig = self.plot_log_log_complexity(prim_res, kruskal_res)
            path = os.path.join(output_dir, '6_scalability_loglog.pdf')
            fig.savefig(path, bbox_inches='tight')
            plt.close(fig)
            saved.append(path)

            # Графік пам'яті (тільки якщо вимірювалась)
            if any(r.peak_memory_kb > 0 for r in prim_res):
                fig = self.plot_memory_scalability(prim_res, kruskal_res)
                path = os.path.join(output_dir, '6b_memory_scalability.pdf')
                fig.savefig(path, bbox_inches='tight')
                plt.close(fig)
                saved.append(path)

        # Графік щільності
        if density_results and 'PrimMST' in density_results and 'KruskalMST' in density_results:
            fig = self.plot_density_impact(
                density_results['PrimMST'], density_results['KruskalMST']
            )
            path = os.path.join(output_dir, '7_density_impact.pdf')
            fig.savefig(path, bbox_inches='tight')
            plt.close(fig)
            saved.append(path)

        # Графіки варіантів куп Пріма
        if prim_variants_res:
            fig = self.plot_prim_variants(prim_variants_res)
            path = os.path.join(output_dir, '9_prim_variants_comparison.pdf')
            fig.savefig(path, bbox_inches='tight')
            plt.close(fig)
            saved.append(path)

        if dary_d_res:
            fig = self.plot_dary_impact(dary_d_res)
            path = os.path.join(output_dir, '10_dary_d_impact.pdf')
            fig.savefig(path, bbox_inches='tight')
            plt.close(fig)
            saved.append(path)

        # Графік масштабованості 2-EC доповнення
        if augmentation_res:
            fig = self.plot_augmentation_scalability(augmentation_res)
            path = os.path.join(output_dir, '11_augmentation_scalability.pdf')
            fig.savefig(path, bbox_inches='tight')
            plt.close(fig)
            saved.append(path)

        # Динамічний графік
        if dynamic_res:
            fig = self.plot_dynamic_vs_static(
                dynamic_res['static_ms'],
                dynamic_res['dynamic_ms'],
                v=dynamic_res.get('node_count', 500),
                static_std_ms=dynamic_res.get('static_std_ms', 0),
                dynamic_std_ms=dynamic_res.get('dynamic_std_ms', 0),
            )
            path = os.path.join(output_dir, '8_dynamic_update.pdf')
            fig.savefig(path, bbox_inches='tight')
            plt.close(fig)
            saved.append(path)

        print(f"Успішно згенеровано {len(saved)} графіків у форматі PDF!")
        return saved


# Скрипт для генерації всього пакету графіків за один запуск
if __name__ == "__main__":
    from .benchmarks import Benchmark

    print("Запуск повного пакету бенчмарків...")
    bench = Benchmark()

    res_comp = bench.full_comparison(
        sizes=[50, 100, 200, 500, 1000], graph_type='random', verbose=True
    )
    res_dens = bench.test_density_impact(node_count=200, verbose=True)
    res_dyn = bench.test_dynamic_vs_static(node_count=500, verbose=True)

    analyzer = ComplexityAnalyzer()
    analyzer.save_all_plots(
        results=res_comp,
        density_results=res_dens,
        dynamic_res=res_dyn,
        output_dir="output_plots"
    )