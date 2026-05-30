"""Покрокова анімація побудови МКД (matplotlib animation, експорт у GIF)."""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyBboxPatch
from typing import List, Optional, Set, Tuple, Dict

from .config import VisualizationConfig, PRESET_ANIMATION
from models import Graph, Node, Edge
from algorithms import MSTResult


class StepByStepVisualizer:
    """
    Покрокова візуалізація побудови МКД.

    Показує:
        - Як алгоритм обирає ребра крок за кроком
        - Відвідані та невідвідані вершини
        - Ребра-кандидати в черзі
        - Поточну вартість МКД

    Example:
        >>> from algorithms import PrimMST
        >>>
        >>> prim = PrimMST(graph, record_steps=True)
        >>> result = prim.find_mst()
        >>>
        >>> animator = StepByStepVisualizer(graph, result)
        >>> animator.animate()  # В реальному часі
        >>> animator.save_gif("prim_animation.gif")  # Зберегти GIF
    """

    def __init__(self,
                 graph: Graph,
                 mst_result: MSTResult,
                 config: Optional[VisualizationConfig] = None):
        """
        Ініціалізує аніматор.

        Args:
            graph: Граф
            mst_result: Результат МКД з кроками (record_steps=True)
            config: Конфігурація візуалізації
        """
        self.graph = graph
        self.mst_result = mst_result
        self.config = config or PRESET_ANIMATION
        self.steps = mst_result.steps

        if not self.steps:
            raise ValueError("MSTResult не містить кроків. "
                             "Використовуйте record_steps=True при створенні алгоритму.")

        # Позиції вершин
        self._positions: Dict[int, Tuple[float, float]] = {}
        self._calculate_positions()

        # Кольори
        self.colors = self.config.colors
        self.sizes = self.config.sizes
        self.labels = self.config.get_labels()

        # Стан анімації
        self.fig: Optional[plt.Figure] = None
        self.ax: Optional[plt.Axes] = None
        self.ax_progress: Optional[plt.Axes] = None
        self._animation: Optional[animation.FuncAnimation] = None

    def _calculate_positions(self) -> None:
        """Обчислює позиції вершин."""
        for node in self.graph.iter_nodes():
            self._positions[node.id] = (node.y, node.x)

    def _get_bounds(self) -> Tuple[float, float, float, float]:
        """Повертає межі графіка."""
        xs = [pos[0] for pos in self._positions.values()]
        ys = [pos[1] for pos in self._positions.values()]

        margin_x = (max(xs) - min(xs)) * 0.15
        margin_y = (max(ys) - min(ys)) * 0.15
        margin_x = max(margin_x, 0.01)
        margin_y = max(margin_y, 0.01)

        return (min(xs) - margin_x, max(xs) + margin_x,
                min(ys) - margin_y, max(ys) + margin_y)

    def _get_visited_nodes(self, step_edges: List[Edge]) -> Set[int]:
        """Повертає відвідані вершини на даному кроці."""
        visited = set()
        for edge in step_edges:
            visited.add(edge.node1)
            visited.add(edge.node2)

        # Додаємо стартову вершину (якщо є ребра)
        if step_edges:
            # Знаходимо стартову вершину з першого кроку
            first_step = self.steps[0]
            if first_step:
                visited.add(first_step[0].node1)
                visited.add(first_step[0].node2)

        return visited

    def _setup_figure(self) -> None:
        """Створює фігуру з двома областями: граф і прогрес-бар."""
        self.fig = plt.figure(
            figsize=(self.sizes.figure_width, self.sizes.figure_height + 1),
            facecolor=self.colors.background
        )

        # Основна область для графа (95% висоти)
        self.ax = self.fig.add_axes([0.05, 0.12, 0.9, 0.82])
        self.ax.set_facecolor(self.colors.background)

        # Область для прогрес-бару (5% висоти)
        self.ax_progress = self.fig.add_axes([0.05, 0.02, 0.9, 0.06])
        self.ax_progress.set_facecolor(self.colors.info_background)

    def _draw_frame(self, step_index: int) -> None:
        """Малює один кадр анімації."""
        self.ax.clear()
        self.ax_progress.clear()

        # Поточні ребра МКД
        if step_index < len(self.steps):
            current_edges = self.steps[step_index]
        else:
            current_edges = self.steps[-1] if self.steps else []

        # Попередні ребра (для визначення нового)
        prev_edges = self.steps[step_index - 1] if step_index > 0 else []

        # Нове ребро на цьому кроці
        new_edge = None
        if len(current_edges) > len(prev_edges):
            new_edge = current_edges[-1]

        # Відвідані вершини
        visited = self._get_visited_nodes(current_edges)

        # Нова вершина
        new_node = None
        if new_edge:
            prev_visited = self._get_visited_nodes(prev_edges)
            for node_id in [new_edge.node1, new_edge.node2]:
                if node_id not in prev_visited:
                    new_node = node_id
                    break

        # 1. Малюємо всі можливі ребра (сірі)
        self._draw_all_edges()

        # 2. Малюємо ребра МКД (зелені)
        self._draw_mst_edges(current_edges, new_edge)

        # 3. Малюємо нове ребро (червоне, поверх)
        if new_edge:
            self._draw_new_edge(new_edge)

        # 4. Малюємо вершини
        self._draw_nodes(visited, new_node)

        # 5. Підписи вершин
        self._draw_labels()

        # 6. Налаштування осей
        bounds = self._get_bounds()
        self.ax.set_xlim(bounds[0], bounds[1])
        self.ax.set_ylim(bounds[2], bounds[3])
        self.ax.set_aspect('equal')
        self.ax.axis('off')

        # 7. Заголовок з інформацією про крок
        self._draw_title(step_index, current_edges, new_edge)

        # 8. Легенда
        self._draw_legend()

        # 9. Прогрес-бар
        self._draw_progress_bar(step_index)

        # 10. Інформаційна панель
        self._draw_info_panel(step_index, current_edges, new_edge)

    def _draw_all_edges(self) -> None:
        """Малює всі можливі ребра (сірі)."""
        for edge in self.graph.iter_edges():
            pos1 = self._positions[edge.node1]
            pos2 = self._positions[edge.node2]

            self.ax.plot(
                [pos1[0], pos2[0]], [pos1[1], pos2[1]],
                color=self.colors.edge_default,
                linewidth=self.sizes.edge_width_default,
                alpha=0.3,
                zorder=1
            )

    def _draw_mst_edges(self, edges: List[Edge], new_edge: Optional[Edge]) -> None:
        """Малює ребра МКД (зелені)."""
        for edge in edges:
            if edge == new_edge:
                continue  # Нове ребро малюємо окремо

            pos1 = self._positions[edge.node1]
            pos2 = self._positions[edge.node2]

            self.ax.plot(
                [pos1[0], pos2[0]], [pos1[1], pos2[1]],
                color=self.colors.edge_mst,
                linewidth=self.sizes.edge_width_mst,
                alpha=0.9,
                zorder=2
            )

    def _draw_new_edge(self, edge: Edge) -> None:
        """Малює нове ребро (червоне) з ефектом."""
        pos1 = self._positions[edge.node1]
        pos2 = self._positions[edge.node2]

        # Світіння (glow effect)
        for width, alpha in [(8, 0.2), (6, 0.3), (4, 0.5)]:
            self.ax.plot(
                [pos1[0], pos2[0]], [pos1[1], pos2[1]],
                color=self.colors.edge_current,
                linewidth=width,
                alpha=alpha,
                zorder=3
            )

        # Основна лінія
        self.ax.plot(
            [pos1[0], pos2[0]], [pos1[1], pos2[1]],
            color=self.colors.edge_current,
            linewidth=self.sizes.edge_width_current,
            alpha=1.0,
            zorder=4
        )

        # Підпис довжини
        mid_x = (pos1[0] + pos2[0]) / 2
        mid_y = (pos1[1] + pos2[1]) / 2

        self.ax.annotate(
            f"{edge.distance:.0f} м",
            (mid_x, mid_y),
            fontsize=self.sizes.font_size_label,
            fontweight='bold',
            color=self.colors.edge_current,
            ha='center',
            va='bottom',
            zorder=10,
            bbox=dict(
                boxstyle='round,pad=0.3',
                facecolor='white',
                edgecolor=self.colors.edge_current,
                alpha=0.9
            )
        )

    def _draw_nodes(self, visited: Set[int], new_node: Optional[int]) -> None:
        """Малює вершини з різними станами."""
        for node_id, pos in self._positions.items():
            if node_id == new_node:
                # Нова вершина — червона з ефектом
                for size, alpha in [(500, 0.2), (400, 0.3)]:
                    self.ax.scatter(
                        pos[0], pos[1],
                        c=self.colors.edge_current,
                        s=size,
                        alpha=alpha,
                        zorder=4
                    )
                self.ax.scatter(
                    pos[0], pos[1],
                    c=self.colors.edge_current,
                    s=self.sizes.node_size_highlighted,
                    edgecolors='white',
                    linewidths=3,
                    zorder=5
                )
            elif node_id in visited:
                # Відвідана вершина — темна
                self.ax.scatter(
                    pos[0], pos[1],
                    c=self.colors.node_default,
                    s=self.sizes.node_size,
                    edgecolors=self.colors.node_border,
                    linewidths=self.sizes.node_border_width,
                    zorder=5
                )
            else:
                # Невідвідана вершина — світла
                self.ax.scatter(
                    pos[0], pos[1],
                    c='white',
                    s=self.sizes.node_size,
                    edgecolors=self.colors.edge_default,
                    linewidths=self.sizes.node_border_width,
                    zorder=5
                )

    def _draw_labels(self) -> None:
        """Малює підписи вершин."""
        for node in self.graph.iter_nodes():
            pos = self._positions[node.id]
            label = node.name if len(node.name) <= 12 else node.name[:10] + "..."

            self.ax.annotate(
                label,
                pos,
                xytext=(0, 14),
                textcoords='offset points',
                fontsize=self.sizes.font_size_label,
                color=self.colors.text_primary,
                ha='center',
                va='bottom',
                fontweight='medium',
                zorder=10
            )

    def _draw_title(self, step_index: int, edges: List[Edge], new_edge: Optional[Edge]) -> None:
        """Малює заголовок з інформацією."""
        total_steps = len(self.steps)
        current_step = step_index + 1  # Для користувача кроки з 1

        if new_edge:
            n1 = self.graph.get_node(new_edge.node1).name
            n2 = self.graph.get_node(new_edge.node2).name
            title = f"Крок {current_step}/{total_steps}: Додаємо {n1} → {n2}"
        else:
            title = f"Крок {current_step}/{total_steps}"

        self.ax.set_title(
            title,
            fontsize=self.sizes.font_size_title,
            fontweight='bold',
            color=self.colors.text_primary,
            pad=15
        )

    def _draw_legend(self) -> None:
        """Малює легенду."""
        from matplotlib.lines import Line2D

        legend_elements = [
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=self.colors.node_default,
                   markeredgecolor=self.colors.node_border,
                   markersize=10, label='Відвідана'),
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor='white',
                   markeredgecolor=self.colors.edge_default,
                   markersize=10, label='Невідвідана'),
            Line2D([0], [0], marker='o', color='w',
                   markerfacecolor=self.colors.edge_current,
                   markersize=10, label='Нова вершина'),
            Line2D([0], [0], color=self.colors.edge_mst,
                   linewidth=3, label='Ребра МКД'),
            Line2D([0], [0], color=self.colors.edge_current,
                   linewidth=3, label='Нове ребро'),
        ]

        self.ax.legend(
            handles=legend_elements,
            loc='upper left',
            fontsize=self.sizes.font_size_legend - 1,
            framealpha=0.9
        )

    def _draw_progress_bar(self, step_index: int) -> None:
        """Малює прогрес-бар."""
        total_steps = len(self.steps)
        progress = (step_index + 1) / total_steps if total_steps > 0 else 0

        self.ax_progress.set_xlim(0, 1)
        self.ax_progress.set_ylim(0, 1)
        self.ax_progress.axis('off')

        # Фон прогрес-бару
        self.ax_progress.add_patch(
            FancyBboxPatch(
                (0.05, 0.2), 0.9, 0.6,
                boxstyle="round,pad=0.02",
                facecolor=self.colors.edge_default,
                edgecolor=self.colors.info_border,
                alpha=0.3
            )
        )

        # Заповнений прогрес
        if progress > 0:
            self.ax_progress.add_patch(
                FancyBboxPatch(
                    (0.05, 0.2), 0.9 * progress, 0.6,
                    boxstyle="round,pad=0.02",
                    facecolor=self.colors.edge_mst,
                    edgecolor='none',
                    alpha=0.8
                )
            )

        # Текст прогресу
        self.ax_progress.text(
            0.5, 0.5,
            f"{self.mst_result.algorithm_name}  •  Крок {step_index + 1} з {total_steps}",
            ha='center', va='center',
            fontsize=self.sizes.font_size_label,
            fontweight='bold',
            color=self.colors.text_primary
        )

    def _draw_info_panel(self, step_index: int, edges: List[Edge], new_edge: Optional[Edge]) -> None:
        """Малює інформаційну панель."""
        current_cost = sum(e.weight for e in edges)
        current_distance = sum(e.distance for e in edges)

        info_text = (
            f"Ребер: {len(edges)}\n"
            f"Довжина: {current_distance:,.0f} м\n"
            f"Вартість: {current_cost:,.0f} грн"
        )

        self.ax.text(
            0.98, 0.02,
            info_text,
            transform=self.ax.transAxes,
            fontsize=self.sizes.font_size_info,
            fontfamily='monospace',
            va='bottom', ha='right',
            bbox=dict(
                boxstyle='round,pad=0.5',
                facecolor=self.colors.info_background,
                edgecolor=self.colors.info_border,
                alpha=0.9
            ),
            zorder=10
        )

    def animate(self, interval: Optional[int] = None, repeat: bool = False) -> None:
        """
        Запускає анімацію в реальному часі.

        Args:
            interval: Час між кадрами (мс). None = з конфігурації
            repeat: Повторювати анімацію
        """
        self._setup_figure()

        interval = interval or self.config.export.animation_interval

        def update(frame):
            self._draw_frame(frame)
            return []

        # Додаємо початковий кадр (порожній граф)
        total_frames = len(self.steps)

        self._animation = animation.FuncAnimation(
            self.fig,
            update,
            frames=total_frames,
            interval=interval,
            repeat=repeat,
            blit=False
        )

        plt.show()

    def save_gif(self, path: str, interval: Optional[int] = None, dpi: int = 100) -> None:
        """
        Зберігає анімацію як GIF.

        Args:
            path: Шлях до файлу
            interval: Час між кадрами (мс)
            dpi: Роздільність
        """
        self._setup_figure()

        interval = interval or self.config.export.animation_interval

        def update(frame):
            self._draw_frame(frame)
            return []

        anim = animation.FuncAnimation(
            self.fig,
            update,
            frames=len(self.steps),
            interval=interval,
            blit=False
        )

        print(f"⏳ Зберігаю GIF ({len(self.steps)} кадрів)...")
        fps = max(1, 1000 // interval)
        anim.save(path, writer='pillow', fps=fps, dpi=dpi)
        print(f"✅ Збережено: {path}")

        plt.close(self.fig)

    def step_through(self) -> None:
        """
        Інтерактивний режим: натисни Enter для наступного кроку.
        """
        self._setup_figure()

        print("\n🎬 Інтерактивний режим")
        print("   Натисни Enter для наступного кроку")
        print("   Введи 'q' для виходу\n")

        for i in range(len(self.steps)):
            self._draw_frame(i)
            self.fig.canvas.draw()
            plt.pause(0.01)

            user_input = input(f"   Крок {i + 1}/{len(self.steps)} > ")
            if user_input.lower() == 'q':
                break

        print("\n✅ Завершено!")
        plt.show()