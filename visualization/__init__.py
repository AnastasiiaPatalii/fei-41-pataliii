"""
Модуль visualization — Візуалізація графів та МКД

Експортує:
    - StaticVisualizer: Статична візуалізація графа
    - StepByStepVisualizer: Покрокова анімація
    - plot_comparison: Порівняння кількох алгоритмів
    - VisualizationConfig: Налаштування візуалізації
    - Пресети: PRESET_DEFAULT, PRESET_PRINT, PRESET_PRESENTATION, PRESET_ANIMATION
"""

from .config import (
    VisualizationConfig,
    ColorScheme,
    SizeConfig,
    ExportConfig,
    PRESET_DEFAULT,
    PRESET_PRINT,
    PRESET_PRESENTATION,
    PRESET_ANIMATION
)
from .static_plot import StaticVisualizer, plot_comparison
from .step_by_step import StepByStepVisualizer

__all__ = [
    'StaticVisualizer',
    'StepByStepVisualizer',
    'plot_comparison',
    'VisualizationConfig',
    'ColorScheme',
    'SizeConfig',
    'ExportConfig',
    'PRESET_DEFAULT',
    'PRESET_PRINT',
    'PRESET_PRESENTATION',
    'PRESET_ANIMATION',
]