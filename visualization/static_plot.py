"""Статична візуалізація графа і МКД через matplotlib."""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from typing import List, Optional, Tuple, Dict
import math

from .config import VisualizationConfig, PRESET_DEFAULT
from models import Graph, Node, Edge
from algorithms import MSTResult


class StaticVisualizer:

    def __init__(self,
                 graph: Graph,
                 mst_result: Optional[MSTResult] = None,
                 config: Optional[VisualizationConfig] = None,
                 reserve_edges: Optional[List] = None):
        """reserve_edges — список Edge з 2-EC доповнення (малюються пунктиром)."""
        self.graph = graph
        self.mst_result = mst_result
        self.config = config or PRESET_DEFAULT
        self.reserve_edges = reserve_edges or []

        # Кешуємо позиції вершин
        self._positions: Dict[int, Tuple[float, float]] = {}
        self._calculate_positions()

        # Множина ребер МКД для швидкої перевірки
        self._mst_edges_set: set = set()
        if mst_result:
            for e in mst_result.edges:
                self._mst_edges_set.add((e.node1, e.node2))
                self._mst_edges_set.add((e.node2, e.node1))

        # Фігура та осі
        self.fig: Optional[plt.Figure] = None
        self.ax: Optional[plt.Axes] = None

    def _calculate_positions(self) -> None:
        """Обчислює позиції вершин на основі їх координат."""
        for node in self.graph.iter_nodes():
            # Використовуємо y (довготу) як x-координату,
            # x (широту) як y-координату для географічно коректного вигляду
            self._positions[node.id] = (node.y, node.x)

    def _get_bounds(self) -> Tuple[float, float, float, float]:
        """Повертає межі для осей (min_x, max_x, min_y, max_y)."""
        if not self._positions:
            return (0, 1, 0, 1)

        xs = [pos[0] for pos in self._positions.values()]
        ys = [pos[1] for pos in self._positions.values()]

        margin_x = (max(xs) - min(xs)) * self.config.sizes.margin
        margin_y = (max(ys) - min(ys)) * self.config.sizes.margin

        # Мінімальний відступ, якщо точки дуже близько
        margin_x = max(margin_x, 0.01)
        margin_y = max(margin_y, 0.01)

        return (
            min(xs) - margin_x,
            max(xs) + margin_x,
            min(ys) - margin_y,
            max(ys) + margin_y
        )

    def _is_mst_edge(self, node1: int, node2: int) -> bool:
        """Перевіряє, чи ребро належить МКД."""
        return (node1, node2) in self._mst_edges_set

    def plot(self,
             show_all_edges: bool = True,
             show_mst: bool = True,
             highlight_nodes: Optional[List[int]] = None,
             title: Optional[str] = None) -> plt.Figure:
        """
        Створює візуалізацію графа.

        Args:
            show_all_edges: Показувати всі можливі ребра
            show_mst: Виділяти ребра МКД
            highlight_nodes: Список ID вершин для виділення
            title: Заголовок (якщо None — використовується з конфігурації)

        Returns:
            Matplotlib Figure об'єкт
        """
        colors = self.config.colors
        sizes = self.config.sizes
        labels = self.config.get_labels()

        # Створюємо фігуру
        self.fig, self.ax = plt.subplots(
            figsize=(sizes.figure_width, sizes.figure_height),
            facecolor=colors.background
        )
        self.ax.set_facecolor(colors.background)

        # Малюємо ребра
        if show_all_edges:
            self._draw_edges(show_mst)
        elif show_mst and self.mst_result:
            self._draw_mst_edges_only()

        # Резервні ребра (2-edge-connected) — поверх МКД
        self._draw_reserve_edges()

        # Малюємо вершини
        self._draw_nodes(highlight_nodes)

        # Підписи вершин
        if self.config.show_node_labels:
            self._draw_labels()

        # Налаштування осей
        bounds = self._get_bounds()
        self.ax.set_xlim(bounds[0], bounds[1])
        self.ax.set_ylim(bounds[2], bounds[3])

        # Прибираємо осі для чистішого вигляду
        self.ax.set_aspect('equal')
        self.ax.axis('off')

        # Заголовок
        if title is None:
            title = labels['title']
            if self.mst_result:
                title += f" — {labels['subtitle_mst']}"

        self.ax.set_title(
            title,
            fontsize=sizes.font_size_title,
            fontweight='bold',
            color=colors.text_primary,
            pad=20
        )

        # Легенда
        self._draw_legend(show_all_edges, show_mst)

        # Інформаційна панель
        if self.config.show_info_panel and self.mst_result:
            self._draw_info_panel()

        plt.tight_layout()

        return self.fig

    def _draw_edges(self, show_mst: bool = True) -> None:
        """Малює всі ребра графа."""
        colors = self.config.colors
        sizes = self.config.sizes

        # Спочатку малюємо звичайні ребра (під МКД)
        for edge in self.graph.iter_edges():
            if show_mst and self._is_mst_edge(edge.node1, edge.node2):
                continue  # МКД намалюємо окремо зверху

            pos1 = self._positions[edge.node1]
            pos2 = self._positions[edge.node2]

            self.ax.plot(
                [pos1[0], pos2[0]],
                [pos1[1], pos2[1]],
                color=colors.edge_default,
                linewidth=sizes.edge_width_default,
                alpha=0.4,
                zorder=1
            )

        # Потім малюємо ребра МКД (зверху)
        if show_mst and self.mst_result:
            for edge in self.mst_result.edges:
                pos1 = self._positions[edge.node1]
                pos2 = self._positions[edge.node2]

                self.ax.plot(
                    [pos1[0], pos2[0]],
                    [pos1[1], pos2[1]],
                    color=colors.edge_mst,
                    linewidth=sizes.edge_width_mst,
                    alpha=0.9,
                    zorder=2
                )

                # Підпис довжини ребра
                if self.config.show_edge_distances:
                    mid_x = (pos1[0] + pos2[0]) / 2
                    mid_y = (pos1[1] + pos2[1]) / 2

                    self.ax.annotate(
                        f"{edge.distance:.0f} м",
                        (mid_x, mid_y),
                        fontsize=sizes.font_size_label - 1,
                        color=colors.text_secondary,
                        ha='center',
                        va='bottom',
                        zorder=5,
                        bbox=dict(
                            boxstyle='round,pad=0.2',
                            facecolor='white',
                            edgecolor='none',
                            alpha=0.7
                        )
                    )

    def _draw_mst_edges_only(self) -> None:
        """Малює тільки ребра МКД."""
        colors = self.config.colors
        sizes = self.config.sizes

        for edge in self.mst_result.edges:
            pos1 = self._positions[edge.node1]
            pos2 = self._positions[edge.node2]

            self.ax.plot(
                [pos1[0], pos2[0]],
                [pos1[1], pos2[1]],
                color=colors.edge_mst,
                linewidth=sizes.edge_width_mst,
                alpha=0.9,
                zorder=2
            )

    def _draw_reserve_edges(self) -> None:
        """Резервні (2-EC) ребра — оранжевий пунктир поверх МКД."""
        if not self.reserve_edges:
            return
        sizes = self.config.sizes
        reserve_color = '#FF8C00'
        for edge in self.reserve_edges:
            pos1 = self._positions[edge.node1]
            pos2 = self._positions[edge.node2]
            self.ax.plot(
                [pos1[0], pos2[0]],
                [pos1[1], pos2[1]],
                color=reserve_color,
                linewidth=sizes.edge_width_mst * 0.8,
                alpha=0.85,
                linestyle='--',
                dashes=(5, 3),
                zorder=3
            )

    def _draw_nodes(self, highlight_nodes: Optional[List[int]] = None) -> None:
        """Малює вершини графа."""
        colors = self.config.colors
        sizes = self.config.sizes

        highlight_set = set(highlight_nodes) if highlight_nodes else set()

        for node_id, pos in self._positions.items():
            is_highlighted = node_id in highlight_set

            # Колір та розмір
            color = colors.node_highlight if is_highlighted else colors.node_default
            size = sizes.node_size_highlighted if is_highlighted else sizes.node_size

            self.ax.scatter(
                pos[0], pos[1],
                c=color,
                s=size,
                edgecolors=colors.node_border,
                linewidths=sizes.node_border_width,
                zorder=4
            )

    def _draw_labels(self) -> None:
        """Малює підписи вершин."""
        colors = self.config.colors
        sizes = self.config.sizes

        for node in self.graph.iter_nodes():
            pos = self._positions[node.id]

            # Скорочуємо назву для компактності
            label = node.name
            if len(label) > 15:
                label = label[:12] + "..."

            self.ax.annotate(
                label,
                pos,
                xytext=(0, 12),
                textcoords='offset points',
                fontsize=sizes.font_size_label,
                color=colors.text_primary,
                ha='center',
                va='bottom',
                fontweight='medium',
                zorder=6
            )

    def _draw_legend(self, show_all_edges: bool, show_mst: bool) -> None:
        """Малює легенду."""
        colors = self.config.colors
        sizes = self.config.sizes
        labels = self.config.get_labels()

        legend_elements = []

        # Вершини
        legend_elements.append(
            Line2D([0], [0],
                   marker='o',
                   color='w',
                   markerfacecolor=colors.node_default,
                   markeredgecolor=colors.node_border,
                   markersize=10,
                   label=labels['legend_nodes'])
        )

        # Звичайні ребра
        if show_all_edges:
            legend_elements.append(
                Line2D([0], [0],
                       color=colors.edge_default,
                       linewidth=sizes.edge_width_default + 1,
                       alpha=0.6,
                       label=labels['legend_edges'])
            )

        # Ребра МКД
        if show_mst and self.mst_result:
            legend_elements.append(
                Line2D([0], [0],
                       color=colors.edge_mst,
                       linewidth=sizes.edge_width_mst,
                       label=labels['legend_mst'])
            )

        # Резервні ребра (2-edge-connected)
        if self.reserve_edges:
            legend_elements.append(
                Line2D([0], [0],
                       color='#FF8C00',
                       linewidth=sizes.edge_width_mst * 0.8,
                       linestyle='--',
                       dashes=(5, 3),
                       label=f'Резервні лінії ({len(self.reserve_edges)})')
            )

        self.ax.legend(
            handles=legend_elements,
            loc='upper left',
            fontsize=sizes.font_size_legend,
            framealpha=0.9,
            edgecolor=colors.info_border
        )

    def _draw_info_panel(self) -> None:
        """Малює інформаційну панель з метриками."""
        colors = self.config.colors
        sizes = self.config.sizes
        labels = self.config.get_labels()

        # Формуємо текст
        info_lines = [
            f"{labels['info_algorithm']}: {self.mst_result.algorithm_name}",
            f"{labels['info_edges']}: {self.mst_result.edge_count}",
            f"{labels['info_distance']}: {self.mst_result.total_distance:,.0f} {labels['meters']}",
            f"{labels['info_cost']}: {self.mst_result.total_cost:,.0f} {labels['currency']}",
            f"{labels['info_time']}: {self.mst_result.execution_time * 1000:.2f} {labels['milliseconds']}",
        ]

        info_text = '\n'.join(info_lines)

        # Розміщуємо в правому нижньому куті
        self.ax.text(
            0.98, 0.02,
            info_text,
            transform=self.ax.transAxes,
            fontsize=sizes.font_size_info,
            fontfamily='monospace',
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(
                boxstyle='round,pad=0.5',
                facecolor=colors.info_background,
                edgecolor=colors.info_border,
                alpha=0.9
            ),
            zorder=10
        )

    def save(self,
             path: str,
             dpi: Optional[int] = None,
             transparent: Optional[bool] = None) -> None:
        """
        Зберігає візуалізацію у файл.

        Args:
            path: Шлях до файлу (підтримує .png, .pdf, .svg)
            dpi: Роздільність (якщо None — з конфігурації)
            transparent: Прозорий фон (якщо None — з конфігурації)
        """
        if self.fig is None:
            self.plot()

        export = self.config.export

        self.fig.savefig(
            path,
            dpi=dpi or export.dpi,
            transparent=transparent if transparent is not None else export.transparent,
            bbox_inches=export.bbox_inches,
            facecolor=self.fig.get_facecolor(),
            edgecolor='none'
        )

        print(f"✅ Збережено: {path}")

    def show(self) -> None:
        """Показує візуалізацію на екрані."""
        if self.fig is None:
            self.plot()
        plt.show()

    def close(self) -> None:
        """Закриває фігуру для звільнення пам'яті."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None


def plot_comparison(graph: Graph,
                    results: List[MSTResult],
                    config: Optional[VisualizationConfig] = None) -> plt.Figure:
    """
    Створює порівняльну візуалізацію кількох алгоритмів.

    Args:
        graph: Граф
        results: Список результатів різних алгоритмів
        config: Конфігурація

    Returns:
        Matplotlib Figure з кількома subplot'ами
    """
    config = config or PRESET_DEFAULT
    n = len(results)

    fig, axes = plt.subplots(1, n, figsize=(7 * n, 8))

    if n == 1:
        axes = [axes]

    for ax, result in zip(axes, results):
        viz = StaticVisualizer(graph, result, config)
        viz.ax = ax
        viz.fig = fig

        # Малюємо на відповідній осі
        viz._draw_edges(show_mst=True)
        viz._draw_nodes(None)
        if config.show_node_labels:
            viz._draw_labels()

        bounds = viz._get_bounds()
        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[2], bounds[3])
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_title(
            result.algorithm_name,
            fontsize=config.sizes.font_size_title,
            fontweight='bold'
        )

    plt.tight_layout()
    return fig