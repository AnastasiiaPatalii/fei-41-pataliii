"""Налаштування візуалізації: палітра, розміри, шрифти, параметри експорту."""

from dataclasses import dataclass, field
from typing import Tuple, Dict, Any


@dataclass
class ColorScheme:
    """Кольорова схема для візуалізації."""

    # Вершини (підстанції)
    node_default: str = "#2C3E50"  # Темно-синій
    node_highlight: str = "#E74C3C"  # Червоний (виділення)
    node_border: str = "#1A252F"  # Темна обводка

    # Ребра
    edge_default: str = "#BDC3C7"  # Світло-сірий (всі можливі)
    edge_mst: str = "#27AE60"  # Зелений (ребра МКД)
    edge_current: str = "#E74C3C"  # Червоний (поточне ребро)
    edge_considered: str = "#F39C12"  # Помаранчевий (розглядається)

    # Фон та текст
    background: str = "#FFFFFF"  # Білий фон
    text_primary: str = "#2C3E50"  # Основний текст
    text_secondary: str = "#7F8C8D"  # Другорядний текст

    # Інформаційна панель
    info_background: str = "#ECF0F1"  # Світло-сірий
    info_border: str = "#BDC3C7"  # Сіра рамка


@dataclass
class SizeConfig:
    """Розміри елементів візуалізації."""

    # Розмір фігури (дюйми)
    figure_width: float = 14
    figure_height: float = 10

    # Вершини
    node_size: int = 300  # Базовий розмір
    node_size_highlighted: int = 400  # Виділена вершина
    node_border_width: float = 2.0

    # Ребра
    edge_width_default: float = 1.0  # Звичайні ребра
    edge_width_mst: float = 3.0  # Ребра МКД
    edge_width_current: float = 4.0  # Поточне ребро

    # Шрифти
    font_size_title: int = 16
    font_size_label: int = 9
    font_size_legend: int = 10
    font_size_info: int = 11

    # Відступи
    margin: float = 0.1  # Відступ від країв (частка)


@dataclass
class ExportConfig:
    """Параметри експорту зображень."""

    dpi: int = 150  # Роздільність (150 для екрану, 300 для друку)
    format: str = "png"  # Формат за замовчуванням
    transparent: bool = False  # Прозорий фон
    bbox_inches: str = "tight"  # Обрізка полів

    # Для анімації
    animation_interval: int = 1000  # Мілісекунди між кадрами
    animation_fps: int = 1  # Кадрів на секунду для GIF


@dataclass
class VisualizationConfig:
    """Головний клас конфігурації візуалізації."""

    colors: ColorScheme = field(default_factory=ColorScheme)
    sizes: SizeConfig = field(default_factory=SizeConfig)
    export: ExportConfig = field(default_factory=ExportConfig)

    # Додаткові параметри
    show_edge_weights: bool = False  # Показувати ваги ребер
    show_edge_distances: bool = True  # Показувати довжини
    show_node_labels: bool = True  # Показувати підписи вершин
    show_grid: bool = False  # Показувати сітку
    show_info_panel: bool = True  # Показувати інфо-панель

    # Українська локалізація
    use_ukrainian: bool = True

    def get_labels(self) -> Dict[str, str]:
        """Повертає підписи для графіка."""
        if self.use_ukrainian:
            return {
                'title': 'Оптимізація енергомережі',
                'subtitle_full': 'Повний граф можливих з\'єднань',
                'subtitle_mst': 'Мінімальне кістякове дерево',
                'legend_nodes': 'Підстанції',
                'legend_edges': 'Можливі лінії',
                'legend_mst': 'Оптимальні лінії (МКД)',
                'legend_current': 'Поточне ребро',
                'info_cost': 'Вартість',
                'info_distance': 'Довжина',
                'info_edges': 'Ліній',
                'info_algorithm': 'Алгоритм',
                'info_time': 'Час',
                'currency': 'грн',
                'meters': 'м',
                'milliseconds': 'мс',
            }
        else:
            return {
                'title': 'Power Grid Optimization',
                'subtitle_full': 'Complete graph of possible connections',
                'subtitle_mst': 'Minimum Spanning Tree',
                'legend_nodes': 'Substations',
                'legend_edges': 'Possible lines',
                'legend_mst': 'Optimal lines (MST)',
                'legend_current': 'Current edge',
                'info_cost': 'Cost',
                'info_distance': 'Distance',
                'info_edges': 'Edges',
                'info_algorithm': 'Algorithm',
                'info_time': 'Time',
                'currency': 'UAH',
                'meters': 'm',
                'milliseconds': 'ms',
            }


# Готові пресети для різних випадків
PRESET_DEFAULT = VisualizationConfig()

PRESET_PRINT = VisualizationConfig(
    sizes=SizeConfig(figure_width=12, figure_height=9, font_size_title=14),
    export=ExportConfig(dpi=300, format="pdf")
)

PRESET_PRESENTATION = VisualizationConfig(
    sizes=SizeConfig(figure_width=16, figure_height=10, node_size=400, font_size_title=20),
    export=ExportConfig(dpi=150)
)

PRESET_ANIMATION = VisualizationConfig(
    export=ExportConfig(animation_interval=800, animation_fps=2)
)