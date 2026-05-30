"""Веб-інтерфейс (Dash) для побудови і дослідження МКД."""

import dash
from dash import dcc, html, Input, Output, State, callback_context, no_update, ALL, dash_table
import dash_bootstrap_components as dbc
import dash_leaflet as dl
import plotly.graph_objects as go
import json
import math
import os
import tempfile
import base64
import time

# Імпорт бізнес-логіки
from models import Node, Edge, Graph
from algorithms import (
    PrimMST, PrimMSTDAry, PrimMSTIndexed, KruskalMST, MSTResult,
    augment_to_2_edge_connected, find_bridges,
)
from data.generator import GraphGenerator
from data.database import DatabaseManager
from data.loader import DataLoader
from data.demo import create_demo_graph
from analysis.benchmarks import Benchmark

# ІНІЦІАЛІЗАЦІЯ ДОДАТКУ ТА КОЛЬОРИ
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY], suppress_callback_exceptions=True)
app.title = "Оптимізація енергомережі (МКД)"

COLORS = {
    'node': '#2C3E50',
    'node_border': '#1A252F',
    'node_new': '#F39C12',
    'edge_bg': '#BDC3C7',
    'edge_mst': '#27AE60',
    'edge_new': '#E74C3C',
    'edge_reserve': '#FF8C00',  # яскраво-оранжевий пунктир — резервні ребра
    'prim': '#1F77B4',
    'kruskal': '#FF7F0E',
}

# Межі України для карти редактора
UKRAINE_BOUNDS = [[44.0, 22.0], [52.5, 40.5]]


# ДОПОМІЖНІ ФУНКЦІЇ СЕРІАЛІЗАЦІЇ
def graph_to_dict(graph: Graph) -> dict:
    if not graph:
        return {"nodes": [], "edges": []}
    return {
        "nodes": [n.to_dict() for n in graph.iter_nodes()],
        "edges": [e.to_dict() for e in graph.iter_edges()]
    }

def dict_to_graph(data: dict) -> Graph:
    g = Graph()
    if not data:
        return g
    for n_data in data.get("nodes", []):
        g.add_node(Node.from_dict(n_data))
    for e_data in data.get("edges", []):
        g.add_edge(Edge(e_data['node1'], e_data['node2'], e_data['distance'], e_data.get('cost_per_meter', 150.0)))
    return g

def mst_to_dict(mst_result) -> dict:
    if not mst_result:
        return None
    return {
        "algorithm": mst_result.algorithm_name,
        "total_cost": mst_result.total_cost,
        "total_distance": mst_result.total_distance,
        "execution_time": mst_result.execution_time,
        "peak_memory_kb": mst_result.peak_memory_kb if mst_result.peak_memory_bytes else None,
        "edges": [{"node1": e.node1, "node2": e.node2, "distance": e.distance, "weight": e.weight, "cost_per_meter": e.cost_per_meter} for e in mst_result.edges],
        "steps": [[{"node1": e.node1, "node2": e.node2, "distance": e.distance, "weight": e.weight} for e in step] for step in mst_result.steps]
    }

def get_db_networks():
    try:
        db = DatabaseManager()
        nets = db.list_networks()
        return [{'label': f"{n['name']} ({n['node_count']} ПС)", 'value': n['id']} for n in nets]
    except Exception:
        return []

def run_both_algorithms(graph: Graph):
    """Прім і Крускал на одному графі. Час і пам'ять міряємо окремими
    прогонами, бо tracemalloc сповільнює."""
    prim_result = PrimMST(graph, record_steps=False).find_mst()
    kruskal_result = KruskalMST(graph, record_steps=False).find_mst()

    prim_mem = PrimMST(graph, record_steps=False).find_mst(track_memory=True)
    kruskal_mem = KruskalMST(graph, record_steps=False).find_mst(track_memory=True)
    prim_result.peak_memory_bytes = prim_mem.peak_memory_bytes
    kruskal_result.peak_memory_bytes = kruskal_mem.peak_memory_bytes

    return prim_result, kruskal_result


def serve_layout():
    """Функція гарантує, що при оновленні сторінки список з БД буде актуальним."""
    stores = html.Div([
        dcc.Store(id='store-main-graph', data={"nodes": [], "edges": []}),
        dcc.Store(id='store-mst-result', data=None),
        dcc.Store(id='store-augmentation', data=None),
        dcc.Store(id='store-editor-nodes', data=[]),
        dcc.Store(id='store-editor-edges', data=[]),
        dcc.Store(id='store-selected-node', data=None),
        dcc.Store(id='store-editing-edge', data=None),
        dcc.Store(id='store-db-network-id', data=None),
    ])

    sidebar = html.Div([
        html.H4("⚡ Джерело Даних", className="fw-bold"),
        html.Hr(),

        html.Label("Створити граф:", className="fw-bold text-primary"),
        dcc.Dropdown(
            id='dropdown-data-source',
            options=[
                {'label': 'Оригінальна Демо-мережа (25 ПС)', 'value': 'demo'},
                {'label': 'Повний граф (Кожен з кожним)', 'value': 'random'},
                {'label': 'Кластерний граф (Міста)', 'value': 'cluster'},
                {'label': 'Сітковий граф (Вулиці)', 'value': 'grid'},
                {'label': 'Розріджений граф', 'value': 'sparse'}
            ],
            value='demo', clearable=False, className="mb-2"
        ),
        html.Div(id='generation-params', children=[
            html.Label("Підстанцій:"),
            dcc.Slider(id='slider-num-nodes', min=5, max=1000, step=5, value=25,
                       marks={10: '10', 100: '100', 200: '200', 500: '500', 1000: '1000'},
                       tooltip={'placement': 'top', 'always_visible': False},
                       className="mb-3"),
            dbc.Row([
                dbc.Col([html.Label("Seed:"), dbc.Input(id='input-seed', type='number', value=42, size="sm")], width=6),
                dbc.Col([html.Label("Грн/м:"), dbc.Input(id='input-cost', type='number', value=150.0, size="sm")], width=6),
            ], className="mb-3"),
            dbc.Button("Згенерувати", id='btn-generate', color="primary", size="sm", className="w-100 mb-3")
        ]),

        html.Hr(),
        html.Label("База Даних:", className="fw-bold text-success"),
        dcc.Dropdown(id='dropdown-db-networks', options=get_db_networks(), placeholder="Оберіть мережу з БД...", className="mb-2"),
        dbc.Row([
            dbc.Col(dbc.Button("Завантажити", id='btn-load-db', color="success", outline=True, size="sm", className="w-100"), width=8),
            dbc.Col(dbc.Button("Видалити", id='btn-delete-db', color="danger", outline=True, size="sm", className="w-100", title="Видалити обрану мережу"), width=4),
        ], className="mb-2"),
        dbc.Input(id='input-save-name',placeholder="Назва для збереження...", size="sm", className="mb-2"),
        dbc.Button("💾 Зберегти поточний в БД", id='btn-save-db', color="success", size="sm", className="w-100 mb-3"),
        html.Div(id='db-alerts'),

        html.Hr(),
        html.Label("Імпорт з файлу:", className="fw-bold text-info"),
        dcc.Upload(
            id='upload-data',
            children=html.Div(['Перетягніть або ', html.A('Оберіть JSON/CSV')]),
            style={'width': '100%', 'height': '40px', 'lineHeight': '40px', 'borderWidth': '1px', 'borderStyle': 'dashed', 'borderRadius': '5px', 'textAlign': 'center', 'cursor': 'pointer'},
            multiple=False, className="mb-3"
        ),
        html.Div(id='upload-alerts'),

        html.Hr(),
        html.Label("Оберіть Алгоритм МКД:", className="fw-bold"),
        dcc.RadioItems(
            id='radio-algorithm',
            options=[
                {'label': ' Пріма (Binary Heap, lazy)', 'value': 'prim_binary'},
                {'label': ' Пріма (4-ary Heap, lazy)', 'value': 'prim_dary'},
                {'label': ' Пріма (Indexed PQ, decrease-key)', 'value': 'prim_indexed'},
                {'label': ' Крускала (Union-Find)', 'value': 'kruskal'},
            ],
            value='prim_binary', inputStyle={"margin-right": "10px"},
            labelStyle={'display': 'block', 'margin-bottom': '4px'}
        ),

    ], style={'padding': '20px', 'backgroundColor': '#f8f9fa', 'height': '100vh', 'overflowY': 'auto', 'borderRight': '1px solid #dee2e6'})

    content = html.Div([
        dbc.Tabs([
            dbc.Tab(label="📊 Граф та МКД", tab_id="tab-graph", children=[
                dbc.Card(dbc.CardBody([
                    dbc.Row([
                        dbc.Col(dbc.Card(dbc.CardBody([html.H6("Алгоритм", className="text-muted mb-1"), html.H5(id='stat-algo', className="fw-bold")]), className="text-center shadow-sm border-0 bg-light")),
                        dbc.Col(dbc.Card(dbc.CardBody([html.H6("Ребер МКД", className="text-muted mb-1"), html.H5(id='stat-edges', className="fw-bold")]), className="text-center shadow-sm border-0 bg-light")),
                        dbc.Col(dbc.Card(dbc.CardBody([html.H6("Загальна Довжина", className="text-primary mb-1"), html.H5(id='stat-dist', className="fw-bold")]), className="text-center shadow-sm border-0 bg-light")),
                        dbc.Col(dbc.Card(dbc.CardBody([html.H6("Загальна Вартість", className="text-success mb-1"), html.H5(id='stat-cost', className="fw-bold")]), className="text-center shadow-sm border-0 bg-light")),
                        dbc.Col(dbc.Card(dbc.CardBody([html.H6("Час Виконання", className="text-danger mb-1"), html.H5(id='stat-time', className="fw-bold")]), className="text-center shadow-sm border-0 bg-light")),
                        dbc.Col(dbc.Card(dbc.CardBody([html.H6("Пам'ять (пік)", className="text-warning mb-1"), html.H5(id='stat-memory', className="fw-bold")]), className="text-center shadow-sm border-0 bg-light")),
                    ], className="mb-3"),
                    dcc.Graph(id='plot-main-graph', style={'height': '55vh'}),

                    html.Hr(),
                    dbc.Row([
                        dbc.Col([dbc.Button("Порівняти Пріма і Крускала на поточному графі", id='btn-compare-algos', color="info", outline=True, className="w-100")], width=4),
                        dbc.Col([dbc.Button("Додати резервування (2-edge-connected)", id='btn-augment-2ec', color="warning", className="w-100")], width=4),
                        dbc.Col([dbc.Button("💾 Зберегти результат МКД в БД", id='btn-save-mst-db', color="success", outline=True, className="w-100")], width=4),
                    ], className="mb-3"),
                    html.Div(id='mst-db-alert'),
                    html.Div(id='augment-results-container'),
                    html.Div(id='compare-results-container'),

                ]), className="mt-3 border-0 shadow-sm")
            ]),

            dbc.Tab(label="✏️ Інтерактивний редактор", tab_id="tab-editor", children=[
                dbc.Row([
                    dbc.Col([
                        dbc.Card(dbc.CardBody([
                            html.Div([
                                dl.Map(id='editor-map', center=[48.46, 35.04], zoom=8,
                                       maxBounds=UKRAINE_BOUNDS, minZoom=6,
                                       children=[
                                    dl.TileLayer(),
                                    dl.LayerGroup(id='editor-map-lines'),
                                    dl.LayerGroup(id='editor-map-markers')
                                ], style={'width': '100%', 'height': '100%'})
                            ], style={'width': '100%', 'height': '55vh', 'borderRadius': '5px', 'overflow': 'hidden', 'border': '2px solid #BDC3C7'}),
                        ]), className="border-0 shadow-sm mb-3"),

                        dbc.Card(dbc.CardBody([
                            html.H5("📊 Властивості ліній (ребер)", className="fw-bold"),
                            html.Div(id='editor-edges-table-container', style={'maxHeight': '30vh', 'overflowY': 'auto'})
                        ]), className="border-0 shadow-sm")

                    ], width=8),
                    dbc.Col([
                        dbc.Card(dbc.CardBody([
                            html.H5("🛠️ Інструмент миші", className="fw-bold"),
                            dcc.RadioItems(
                                id='editor-mouse-mode',
                                options=[
                                    {'label': ' 🖱️ Вказівник (вибір / редагування ліній)', 'value': 'pointer'},
                                    {'label': ' 📍 Додати підстанцію (клік по карті)', 'value': 'add_node'},
                                    {'label': ' 🔗 З\'єднати підстанції (клік по двом ПС)', 'value': 'add_edge'},
                                    {'label': ' 🗑️ Видалити підстанцію (клік по ПС)', 'value': 'delete_node'}
                                ],
                                value='pointer',
                                labelStyle={'display': 'block', 'margin-bottom': '10px'},
                                inputStyle={'margin-right': '10px'}
                            ),
                            html.P("У режимі 'Вказівник' клікніть на лінію на карті.", className="text-muted small mt-2"),
                            html.Div(id='editor-alerts', className="mt-2"),

                            html.Hr(),
                            html.H5("🧨 Швидкі дії", className="fw-bold"),
                            dbc.Button("Очистити всю карту", id='btn-clear-editor', color="danger", outline=True, className="w-100 mb-4"),

                            html.Hr(),
                            html.H5("Синхронізація", className="fw-bold"),
                            dbc.Button("Завантажити поточний граф", id='btn-load-to-editor', color="info", outline=True, className="w-100 mb-2"),
                            dbc.Button("💾 Зберегти зміни як головний граф", id='btn-save-from-editor', color="success", className="w-100"),
                        ]), className="border-0 shadow-sm")
                    ], width=4)
                ], className="mt-3")
            ]),

            dbc.Tab(label="🧮 Розширений редактор", tab_id="tab-adv-editor", children=[
                dbc.Row([
                    dbc.Col([
                        dbc.Card(dbc.CardBody([
                            html.H5("Синхронізація та розрахунки", className="fw-bold"),
                            dbc.Row([
                                dbc.Col(dbc.Button("Завантажити поточний граф", id="btn-adv-load", color="info", outline=True, className="w-100"), width=3),
                                dbc.Col(dbc.Button("💾 Зберегти як головний", id="btn-adv-save", color="success", className="w-100"), width=3),
                                dbc.Col(dcc.Dropdown(
                                    id="dropdown-adv-method",
                                    options=[{'label': 'Гаверсинус (GPS)', 'value': 'haversine'}, {'label': 'Евклідова відстань', 'value': 'euclidean'}],
                                    value='haversine', clearable=False
                                ), width=3),
                                dbc.Col(dbc.Button("Перерахувати відстані", id="btn-adv-recalc", color="warning", className="w-100"), width=3),
                            ]),
                            html.Div(id="adv-alerts", className="mt-2")
                        ]), className="border-0 shadow-sm mb-3")
                    ], width=12)
                ], className="mt-3"),

                dbc.Row([
                    dbc.Col([
                        dbc.Card(dbc.CardBody([
                            html.H5("📍 Підстанції (Вершини)", className="fw-bold"),
                            dbc.Button("Додати рядок", id="btn-adv-add-node", color="secondary", size="sm", className="mb-2"),
                            dash_table.DataTable(
                                id='table-adv-nodes',
                                columns=[
                                    {'name': 'ID', 'id': 'id', 'type': 'numeric'},
                                    {'name': 'Назва', 'id': 'name', 'type': 'text'},
                                    {'name': 'Широта (x)', 'id': 'x', 'type': 'numeric'},
                                    {'name': 'Довгота (y)', 'id': 'y', 'type': 'numeric'}
                                ],
                                data=[], editable=True, row_deletable=True,
                                style_cell={'textAlign': 'left', 'padding': '5px'},
                                style_header={'backgroundColor': '#f8f9fa', 'fontWeight': 'bold'},
                                style_table={'overflowX': 'auto', 'maxHeight': '45vh', 'overflowY': 'auto'}
                            )
                        ]), className="border-0 shadow-sm")
                    ], width=6),
                    dbc.Col([
                        dbc.Card(dbc.CardBody([
                            html.H5("🔗 Лінії (Ребра)", className="fw-bold"),
                            dbc.Button("Додати рядок", id="btn-adv-add-edge", color="secondary", size="sm", className="mb-2"),
                            dash_table.DataTable(
                                id='table-adv-edges',
                                columns=[
                                    {'name': 'ПС 1 (ID)', 'id': 'node1', 'type': 'numeric'},
                                    {'name': 'ПС 2 (ID)', 'id': 'node2', 'type': 'numeric'},
                                    {'name': 'Довжина (м)', 'id': 'distance', 'type': 'numeric'},
                                    {'name': 'Вартість (грн/м)', 'id': 'cost_per_meter', 'type': 'numeric'}
                                ],
                                data=[], editable=True, row_deletable=True,
                                style_cell={'textAlign': 'left', 'padding': '5px'},
                                style_header={'backgroundColor': '#f8f9fa', 'fontWeight': 'bold'},
                                style_table={'overflowX': 'auto', 'maxHeight': '45vh', 'overflowY': 'auto'}
                            )
                        ]), className="border-0 shadow-sm")
                    ], width=6)
                ])
            ]),

            dbc.Tab(label="🎬 Анімація", tab_id="tab-steps", children=[
                dbc.Card(dbc.CardBody([
                    html.H4(id='anim-title', className="text-center mb-3 fw-bold"),
                    dbc.Row([
                        dbc.Col([
                            dbc.ButtonGroup([
                                dbc.Button("|<", id="btn-anim-start", color="secondary", outline=True, size="sm", title="На початок"),
                                dbc.Button("<", id="btn-anim-prev", color="secondary", outline=True, size="sm", title="Попередній крок"),
                                dbc.Button("Відтворити", id="btn-anim-play", color="success", size="sm", style={'minWidth': '140px'}),
                                dbc.Button(">", id="btn-anim-next", color="secondary", outline=True, size="sm", title="Наступний крок"),
                                dbc.Button(">|", id="btn-anim-end", color="secondary", outline=True, size="sm", title="В кінець"),
                            ], className="w-100")
                        ], width=5),
                        dbc.Col([
                            dcc.Slider(id='slider-anim-step', min=1, max=2, step=1, value=1, tooltip={"placement": "top", "always_visible": False})
                        ], width=5, className="d-flex align-items-center"),
                        dbc.Col([
                            html.Label("Швидкість:", className="small text-muted mb-0"),
                            dcc.Dropdown(
                                id='dropdown-anim-speed',
                                options=[
                                    {'label': '0.25×', 'value': 3200},
                                    {'label': '0.5×', 'value': 1600},
                                    {'label': '1×', 'value': 800},
                                    {'label': '2×', 'value': 400},
                                    {'label': '4×', 'value': 200},
                                ],
                                value=800, clearable=False, searchable=False, style={'fontSize': '0.85rem'}
                            )
                        ], width=2),
                    ], className="mb-3 align-items-center"),
                    dcc.Interval(id='anim-interval', interval=800, disabled=True),
                    dcc.Graph(id='plot-anim-graph', style={'height': '58vh'})
                ]), className="mt-3 border-0 shadow-sm")
            ]),

            dbc.Tab(label="⚖️ Бенчмарки", tab_id="tab-bench", children=[
                dbc.Card(dbc.CardBody([
                    html.H5("Аналіз алгоритмів", className="fw-bold"),
                    html.P("Виберіть тип дослідження для запуску на бекенді та побудови інтерактивних графіків.", className="text-muted"),
                    dbc.Row([
                        dbc.Col([
                            dcc.Dropdown(
                                id='dropdown-bench-type',
                                options=[
                                    {'label': '1. Масштабованість та Складність (V=20..200, швидко)', 'value': 'complexity'},
                                    {'label': '2. Вплив щільності графа (Точка перетину, V=150)', 'value': 'density'},
                                    {'label': '3. Ефективність динамічного оновлення МКД (V=300)', 'value': 'dynamic'},
                                    {'label': '4. Тест на розрідженому графі (V=500, E=4000)', 'value': 'reference'},
                                    {'label': '5. Повна масштабованість (V=50..1000) — займає ~2 хв', 'value': 'complexity_full'},
                                    {'label': '6. Порівняння варіантів Пріма (Binary / 4-ary / Indexed PQ)', 'value': 'prim_variants'},
                                ],
                                value='complexity', clearable=False
                            )
                        ], width=8),
                        dbc.Col(dbc.Button("Запустити бенчмарк", id='btn-run-bench', color="danger", className="w-100"), width=4)
                    ], className="mb-4"),
                    dcc.Loading(type="circle", children=[
                        html.Div(id='bench-results-container')
                    ])
                ]), className="mt-3 border-0 shadow-sm")
            ]),

            dbc.Tab(label="💾 Експорт", tab_id="tab-export", children=[
                dbc.Card(dbc.CardBody([
                    html.H5("Зберегти результати на ПК", className="fw-bold mb-4"),
                    dbc.Row([
                        dbc.Col(dbc.Button("Завантажити JSON", id='btn-export-json', color="primary", outline=True, className="w-100"), width=4),
                        dbc.Col(dbc.Button("Завантажити CSV", id='btn-export-csv', color="success", outline=True, className="w-100"), width=4),
                        dbc.Col(dbc.Button("Завантажити GeoJSON", id='btn-export-geojson', color="info", outline=True, className="w-100"), width=4)
                    ]),
                    dcc.Download(id="download-json"),
                    dcc.Download(id="download-csv"),
                    dcc.Download(id="download-geojson")
                ]), className="mt-3 border-0 shadow-sm")
            ])
        ], id="main-tabs", active_tab="tab-graph")
    ], style={'padding': '20px'})

    modal_edit_edge = dbc.Modal([
        dbc.ModalHeader(dbc.ModalTitle("Редагування лінії")),
        dbc.ModalBody([
            html.P(id="modal-edge-nodes-title", className="fw-bold"),
            html.Label("Довжина (метри):"),
            dbc.Input(id="modal-edge-dist", type="number", step=0.1, className="mb-3"),
            html.Label("Вартість прокладання (грн за метр):"),
            dbc.Input(id="modal-edge-cost", type="number", step=0.1, className="mb-3"),
        ]),
        dbc.ModalFooter([
            dbc.Button("Видалити лінію", id="btn-modal-delete-edge", color="danger", className="me-auto"),
            dbc.Button("💾 Зберегти зміни", id="btn-modal-save-edge", color="success")
        ])
    ], id="modal-edit-edge", is_open=False, centered=True)

    return dbc.Container([
        stores,
        modal_edit_edge,
        html.Div(id='dummy-div', style={'display': 'none'}),
        dbc.Row([
            dbc.Col(sidebar, width=3, className="p-0"),
            dbc.Col(content, width=9)
        ])
    ], fluid=True, className="p-0")

# Призначаємо функцію замість статичної змінної
app.layout = serve_layout


# CALLBACKS

app.clientside_callback(
    """
    function(tab) {
        if(tab === 'tab-editor') {
            setTimeout(function() { window.dispatchEvent(new Event('resize')); }, 300);
        }
        return window.dash_clientside.no_update;
    }
    """,
    Output('dummy-div', 'children'),
    Input('main-tabs', 'active_tab')
)

@app.callback(Output('generation-params', 'style'), Input('dropdown-data-source', 'value'))
def toggle_generation_params(source):
    if source == 'demo':
        return {'display': 'none'}
    return {'display': 'block'}

@app.callback(
    Output('store-selected-node', 'data'),
    Input('editor-mouse-mode', 'value'),
    prevent_initial_call=True
)
def reset_selection_on_mode_change(mode):
    return None

# --- ЗАВАНТАЖЕННЯ ГРАФА ---
@app.callback(
    Output('store-main-graph', 'data'),
    Output('upload-alerts', 'children'),
    Output('store-db-network-id', 'data'),
    Input('btn-generate', 'n_clicks'),
    Input('btn-save-from-editor', 'n_clicks'),
    Input('btn-load-db', 'n_clicks'),
    Input('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('dropdown-data-source', 'value'),
    State('slider-num-nodes', 'value'),
    State('input-seed', 'value'),
    State('input-cost', 'value'),
    State('dropdown-db-networks', 'value'),
    State('store-editor-nodes', 'data'),
    State('store-editor-edges', 'data'),
    State('store-db-network-id', 'data'),
    prevent_initial_call=False
)
def handle_graph_loading(btn_gen, btn_save, btn_load_db, file_content, filename,
                         source, num_nodes, seed, cost, db_net_id,
                         ed_nodes, ed_edges, current_db_id):
    ctx = callback_context
    trigger = ctx.triggered[0]['prop_id'] if ctx.triggered else 'init'
    alert = no_update

    if 'btn-save-from-editor' in trigger:
        g = Graph()
        for n in ed_nodes:
            g.add_node(Node(n['id'], n['name'], n['x'], n['y']))
        for e in ed_edges:
            g.add_edge(Edge(e['node1'], e['node2'], e['distance'], e['cost']))
        return graph_to_dict(g), alert, None

    if 'btn-load-db' in trigger:
        if not db_net_id:
            return no_update, dbc.Alert("Оберіть мережу зі списку!", color="warning", duration=3000), no_update
        try:
            db = DatabaseManager()
            loaded_g = db.load_network(db_net_id)
            if loaded_g and loaded_g.node_count() > 0:
                return graph_to_dict(loaded_g), dbc.Alert(
                    f"Завантажено мережу з {loaded_g.node_count()} ПС!", color="success", duration=3000), db_net_id
            else:
                return no_update, dbc.Alert("Мережу не знайдено!", color="danger", duration=3000), no_update
        except Exception as e:
            return no_update, dbc.Alert(f"Помилка БД: {e}", color="danger", duration=3000), no_update

    if 'upload-data' in trigger and file_content:
        content_type, content_string = file_content.split(',')
        decoded = base64.b64decode(content_string)
        ext = ".json" if filename.endswith(".json") else ".csv"
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
                tmp.write(decoded)
                tmp_path = tmp.name
            loaded_g = DataLoader.load(tmp_path, cost_per_meter=cost)
            os.unlink(tmp_path)
            return graph_to_dict(loaded_g), dbc.Alert("Файл успішно завантажено!", color="success", duration=3000), None
        except Exception as e:
            return no_update, dbc.Alert(f"Помилка: {str(e)}", color="danger"), no_update

    if 'btn-generate' in trigger:
        pass
    elif trigger == 'init':
        g = create_demo_graph()
        return graph_to_dict(g), alert, None
    else:
        return no_update, no_update, no_update

    if source == 'demo':
        g = create_demo_graph()
    else:
        generator = GraphGenerator(seed=seed)
        if source == 'random':
            g = generator.random_graph(num_nodes, cost_per_meter=cost, complete=True)
        elif source == 'cluster':
            clusters = max(2, num_nodes // 5)
            g = generator.cluster_graph(clusters, num_nodes // clusters, cost_per_meter=cost)
        elif source == 'grid':
            rows = int(math.sqrt(num_nodes))
            cols = num_nodes // rows if rows > 0 else 1
            g = generator.grid_graph(rows, cols, cost_per_meter=cost)
        elif source == 'sparse':
            g = generator.random_graph(num_nodes, cost_per_meter=cost, complete=False)
            nodes_list = list(g.iter_nodes())
            for i in range(len(nodes_list) - 1):
                n1, n2 = nodes_list[i], nodes_list[i+1]
                if not g.has_edge(n1.id, n2.id):
                    g.add_edge(Edge(n1.id, n2.id, 5000.0, cost))
        else:
            g = create_demo_graph()

    return graph_to_dict(g), alert, None


# --- ЗБЕРЕЖЕННЯ В БД ---
@app.callback(
    Output('db-alerts', 'children'), Output('dropdown-db-networks', 'options'),
    Output('store-db-network-id', 'data', allow_duplicate=True),
    Input('btn-save-db', 'n_clicks'), State('input-save-name', 'value'), State('store-main-graph', 'data'),
    prevent_initial_call=True
)
def save_to_db(n_clicks, name, graph_data):
    if not name or not graph_data['nodes']:
        return dbc.Alert("Введіть назву та завантажте граф!", color="warning", duration=3000), no_update, no_update
    try:
        g = dict_to_graph(graph_data)
        db = DatabaseManager()
        net_id = db.save_network(name, g, cost_per_meter=150.0)
        return (dbc.Alert(f"Мережу '{name}' збережено в БД!", color="success", duration=3000),
                get_db_networks(), net_id)
    except Exception as e:
        return dbc.Alert(f"Помилка БД: {e}", color="danger", duration=4000), no_update, no_update


# --- ВИДАЛЕННЯ З БД ---
@app.callback(
    Output('db-alerts', 'children', allow_duplicate=True),
    Output('dropdown-db-networks', 'options', allow_duplicate=True),
    Output('dropdown-db-networks', 'value'),
    Input('btn-delete-db', 'n_clicks'),
    State('dropdown-db-networks', 'value'),
    prevent_initial_call=True
)
def delete_from_db(n_clicks, db_net_id):
    if not db_net_id:
        return dbc.Alert("Оберіть мережу для видалення!", color="warning", duration=3000), no_update, no_update
    try:
        db = DatabaseManager()
        db.delete_network(db_net_id)
        return (dbc.Alert("Мережу видалено з БД!", color="info", duration=3000),
                get_db_networks(), None)
    except Exception as e:
        return dbc.Alert(f"Помилка видалення: {e}", color="danger", duration=4000), no_update, no_update


# --- ЗБЕРЕЖЕННЯ МКД В БД ---
@app.callback(
    Output('mst-db-alert', 'children'),
    Input('btn-save-mst-db', 'n_clicks'),
    State('store-mst-result', 'data'),
    State('store-main-graph', 'data'),
    State('store-db-network-id', 'data'),
    State('input-save-name', 'value'),
    prevent_initial_call=True
)
def save_mst_to_db(n_clicks, mst_data, graph_data, db_net_id, save_name):
    if not mst_data or not mst_data.get('edges'):
        return dbc.Alert("Спершу побудуйте МКД!", color="warning", duration=3000)

    try:
        db = DatabaseManager()
        if not db_net_id:
            g = dict_to_graph(graph_data)
            net_name = save_name or f"Автозбереження ({len(graph_data['nodes'])} ПС)"
            db_net_id = db.save_network(net_name, g, cost_per_meter=150.0)

        mst_edges = []
        for e in mst_data['edges']:
            mst_edges.append(Edge(e['node1'], e['node2'], e['distance'], e.get('cost_per_meter', 150.0)))

        peak_kb = mst_data.get('peak_memory_kb')
        result = MSTResult(
            edges=mst_edges,
            total_cost=mst_data['total_cost'],
            total_distance=mst_data['total_distance'],
            execution_time=mst_data['execution_time'],
            steps=[],
            algorithm_name=mst_data['algorithm'],
            peak_memory_bytes=int(peak_kb * 1024) if peak_kb else None
        )

        raw_name = mst_data['algorithm']
        if 'Indexed' in raw_name:
            algo_name = "Пріма (Indexed)"
        elif '-ary' in raw_name:
            # "Prim (4-ary Heap)" — d-арна купа. Дефіс відрізняє від "Binary".
            algo_name = "Пріма (d-ary)"
        elif 'Prim' in raw_name:
            algo_name = "Пріма (Binary)"
        else:
            algo_name = "Крускала"
        result_id = db.save_mst_result(db_net_id, result, algorithm=algo_name)

        return dbc.Alert(
            f"✅ Результат МКД ({algo_name}) збережено в БД! "
            f"Мережа ID: {db_net_id}, Результат ID: {result_id}",
            color="success", duration=5000
        )
    except Exception as e:
        return dbc.Alert(f"Помилка збереження МКД: {e}", color="danger", duration=4000)


def _make_algorithm(algorithm_id: str, graph: Graph, record_steps: bool = True):
    """ID з radio-кнопки → готовий екземпляр алгоритму."""
    if algorithm_id == 'prim_binary':
        return PrimMST(graph, record_steps=record_steps)
    if algorithm_id == 'prim_dary':
        return PrimMSTDAry(graph, record_steps=record_steps, d=4)
    if algorithm_id == 'prim_indexed':
        return PrimMSTIndexed(graph, record_steps=record_steps)
    if algorithm_id == 'kruskal':
        return KruskalMST(graph, record_steps=record_steps)
    if algorithm_id == 'prim':  # старий ID зі збереженого store
        return PrimMST(graph, record_steps=record_steps)
    raise ValueError(f"Невідомий алгоритм: {algorithm_id}")


@app.callback(Output('store-mst-result', 'data'), Input('store-main-graph', 'data'), Input('radio-algorithm', 'value'))
def calculate_mst(graph_data, algorithm):
    g = dict_to_graph(graph_data)
    if g.node_count() < 2:
        return None
    # tracemalloc сповільнює, але один UI-запуск і так у мс
    result = _make_algorithm(algorithm, g, record_steps=True).find_mst(track_memory=True)
    return mst_to_dict(result)

# --- ПОРІВНЯННЯ ПРІМА ТА КРУСКАЛА НА ПОТОЧНОМУ ГРАФІ ---
@app.callback(
    Output('compare-results-container', 'children'),
    Input('btn-compare-algos', 'n_clicks'),
    State('store-main-graph', 'data'),
    prevent_initial_call=True
)
def compare_algorithms_on_current(n_clicks, graph_data):
    g = dict_to_graph(graph_data)
    if g.node_count() < 2:
        return dbc.Alert("Завантажте граф для порівняння!", color="warning", duration=3000)

    prim_result, kruskal_result = run_both_algorithms(g)

    if prim_result.execution_time < kruskal_result.execution_time:
        winner = "Пріма"
        ratio = kruskal_result.execution_time / prim_result.execution_time if prim_result.execution_time > 0 else 0
    else:
        winner = "Крускала"
        ratio = prim_result.execution_time / kruskal_result.execution_time if kruskal_result.execution_time > 0 else 0

    comparison_table = dbc.Table([
        html.Thead(html.Tr([
            html.Th("Метрика", style={'width': '40%'}),
            html.Th("Пріма", className="text-center", style={'color': COLORS['prim']}),
            html.Th("Крускала", className="text-center", style={'color': COLORS['kruskal']}),
        ])),
        html.Tbody([
            html.Tr([
                html.Td("Час виконання"),
                html.Td(f"{prim_result.execution_time * 1000:.4f} мс", className="text-center"),
                html.Td(f"{kruskal_result.execution_time * 1000:.4f} мс", className="text-center"),
            ]),
            html.Tr([
                html.Td("Загальна вартість МКД"),
                html.Td(f"{prim_result.total_cost:,.2f} грн", className="text-center"),
                html.Td(f"{kruskal_result.total_cost:,.2f} грн", className="text-center"),
            ]),
            html.Tr([
                html.Td("Загальна довжина"),
                html.Td(f"{prim_result.total_distance / 1000:,.2f} км", className="text-center"),
                html.Td(f"{kruskal_result.total_distance / 1000:,.2f} км", className="text-center"),
            ]),
            html.Tr([
                html.Td("Кількість ребер МКД"),
                html.Td(f"{prim_result.edge_count}", className="text-center"),
                html.Td(f"{kruskal_result.edge_count}", className="text-center"),
            ]),
            html.Tr([
                html.Td("Пам'ять (пік)"),
                html.Td(f"{prim_result.peak_memory_kb:,.1f} КБ" if prim_result.peak_memory_bytes else "—", className="text-center"),
                html.Td(f"{kruskal_result.peak_memory_kb:,.1f} КБ" if kruskal_result.peak_memory_bytes else "—", className="text-center"),
            ]),
        ])
    ], bordered=True, hover=True, size="sm", className="mb-2")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=['Пріма (Binary Heap)'], y=[prim_result.execution_time * 1000],
        marker_color=COLORS['prim'], text=[f"{prim_result.execution_time * 1000:.4f} мс"], textposition='auto'
    ))
    fig.add_trace(go.Bar(
        x=['Крускала (Union-Find)'], y=[kruskal_result.execution_time * 1000],
        marker_color=COLORS['kruskal'], text=[f"{kruskal_result.execution_time * 1000:.4f} мс"], textposition='auto'
    ))
    fig.update_layout(
        title=f"Час виконання на поточному графі (V={g.node_count()}, E={g.edge_count()})",
        yaxis_title="Час (мс)", plot_bgcolor='white', showlegend=False, height=280, margin=dict(l=40, r=40, t=40, b=40)
    )

    cost_match = abs(prim_result.total_cost - kruskal_result.total_cost) < 0.01
    cost_note = "✅ Обидва алгоритми знайшли однакове МКД" if cost_match else "⚠️ Результати різняться (можливо декілька МКД з однаковою вагою)"

    return html.Div([
        html.Hr(),
        html.H5("⚖️ Порівняння алгоритмів на поточному графі", className="fw-bold"),
        dbc.Alert(f"🏆 Переможець: алгоритм {winner} — швидший у {ratio:.1f} разів", color="info", className="fw-bold text-center"),
        dbc.Row([
            dbc.Col(comparison_table, width=6),
            dbc.Col(dcc.Graph(figure=fig, config={'displayModeBar': False}), width=6),
        ]),
        html.P(cost_note, className="text-muted text-center small"),
    ])


# --- ДОПОВНЕННЯ ДО 2-EDGE-CONNECTED ---
@app.callback(
    Output('store-augmentation', 'data'),
    Output('augment-results-container', 'children'),
    Input('btn-augment-2ec', 'n_clicks'),
    State('store-main-graph', 'data'),
    State('store-mst-result', 'data'),
    prevent_initial_call=True
)
def run_augmentation(n_clicks, graph_data, mst_data):
    g = dict_to_graph(graph_data)
    if g.node_count() < 3:
        return None, dbc.Alert(
            "Для 2-edge-connected потрібно щонайменше 3 підстанції!",
            color="warning", duration=4000
        )
    if not mst_data or not mst_data.get('edges'):
        return None, dbc.Alert(
            "Спершу побудуйте МКД, потім додавайте резервування!",
            color="warning", duration=4000
        )

    mst_edges = [
        Edge(e['node1'], e['node2'], e['distance'], e.get('cost_per_meter', 150.0))
        for e in mst_data['edges']
    ]
    result = augment_to_2_edge_connected(g, mst_edges, track_memory=True)

    aug_store = {
        'reserve_edges': [
            {'node1': e.node1, 'node2': e.node2,
             'distance': e.distance, 'cost_per_meter': e.cost_per_meter,
             'weight': e.weight}
            for e in result.reserve_edges
        ],
        'edges_added': result.edges_added,
        'lower_bound': result.lower_bound,
        'approximation_ratio': result.approximation_ratio,
        'reserve_cost': result.reserve_cost,
        'reserve_distance': result.reserve_distance,
        'execution_time': result.execution_time,
        'peak_memory_kb': result.peak_memory_kb,
        'leaves_count': result.leaves_count,
        'bridges_remaining': result.bridges_remaining,
        'is_2_edge_connected': result.is_2_edge_connected,
    }

    mst_cost = mst_data.get('total_cost', 0)
    pct = (result.reserve_cost / mst_cost * 100) if mst_cost else 0

    summary = html.Div([
        html.Hr(),
        html.H5("Резервування мережі (2-edge-connected)", className="fw-bold"),
        dbc.Alert(
            f"✅ Граф став 2-edge-connected — будь-яка лінія може вийти з ладу без розпаду мережі"
            if result.is_2_edge_connected else
            f"⚠️ Залишилось {result.bridges_remaining} вразливих місць (не вдалося покрити)",
            color="success" if result.is_2_edge_connected else "warning",
            className="fw-bold text-center"
        ),
        dbc.Row([
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("Резервних ребер", className="text-muted mb-1"),
                html.H5(f"{result.edges_added} / {result.lower_bound} (опт.)", className="fw-bold text-warning"),
            ]), className="text-center shadow-sm border-0")),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("Апроксимація", className="text-muted mb-1"),
                html.H5(f"{result.approximation_ratio:.2f}x", className="fw-bold"),
            ]), className="text-center shadow-sm border-0")),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("Додатк. вартість", className="text-muted mb-1"),
                html.H5(f"{result.reserve_cost:,.0f} ₴", className="fw-bold text-danger"),
                html.Small(f"+{pct:.1f}% від МКД", className="text-muted"),
            ]), className="text-center shadow-sm border-0")),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("Додатк. довжина", className="text-muted mb-1"),
                html.H5(f"{result.reserve_distance/1000:,.1f} км", className="fw-bold"),
            ]), className="text-center shadow-sm border-0")),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("Час", className="text-muted mb-1"),
                html.H5(f"{result.execution_time*1000:.2f} мс", className="fw-bold text-danger"),
            ]), className="text-center shadow-sm border-0")),
            dbc.Col(dbc.Card(dbc.CardBody([
                html.H6("Пам'ять (пік)", className="text-muted mb-1"),
                html.H5(f"{result.peak_memory_kb:,.1f} КБ", className="fw-bold text-warning"),
            ]), className="text-center shadow-sm border-0")),
        ], className="mb-3"),
        html.P(
            f"Листків у МКД: {result.leaves_count}. "
            f"Нижня теоретична межа: ⌈{result.leaves_count}/2⌉ = {result.lower_bound} ребер.",
            className="text-muted small text-center"
        ),
    ])
    return aug_store, summary


# Очищаємо резервування при зміні графа чи алгоритму
@app.callback(
    Output('store-augmentation', 'data', allow_duplicate=True),
    Output('augment-results-container', 'children', allow_duplicate=True),
    Input('store-mst-result', 'data'),
    prevent_initial_call=True
)
def reset_augmentation(_):
    return None, None


# --- ГОЛОВНИЙ ГРАФ ТА МЕТРИКИ ---
@app.callback(
    Output('plot-main-graph', 'figure'),
    Output('stat-algo', 'children'), Output('stat-edges', 'children'),
    Output('stat-dist', 'children'), Output('stat-cost', 'children'),
    Output('stat-time', 'children'), Output('stat-memory', 'children'),
    Input('store-main-graph', 'data'), Input('store-mst-result', 'data'),
    Input('store-augmentation', 'data')
)
def update_main_view(graph_data, mst_data, aug_data):
    g = dict_to_graph(graph_data)
    fig = go.Figure()
    if g.node_count() == 0:
        fig.update_layout(title="Граф порожній", template="plotly_white")
        return fig, "-", "-", "-", "-", "-", "-"

    positions = {n.id: (n.y, n.x) for n in g.iter_nodes()}

    if g.edge_count() < 1000:
        bg_x, bg_y = [], []
        mst_keys = {(min(e['node1'], e['node2']), max(e['node1'], e['node2'])) for e in mst_data['edges']} if mst_data else set()
        for e in g.iter_edges():
            k = (min(e.node1, e.node2), max(e.node1, e.node2))
            if k not in mst_keys:
                bg_x += [positions[e.node1][0], positions[e.node2][0], None]
                bg_y += [positions[e.node1][1], positions[e.node2][1], None]
        if bg_x:
            fig.add_trace(go.Scatter(x=bg_x, y=bg_y, mode='lines', line=dict(color=COLORS['edge_bg'], width=1), opacity=0.3, hoverinfo='none'))

    if mst_data and mst_data['edges']:
        mst_x, mst_y, hover_text, hover_x, hover_y = [], [], [], [], []
        for e in mst_data['edges']:
            p1, p2 = positions[e['node1']], positions[e['node2']]
            mst_x += [p1[0], p2[0], None]; mst_y += [p1[1], p2[1], None]
            hover_x.append((p1[0] + p2[0])/2); hover_y.append((p1[1] + p2[1])/2)
            hover_text.append(f"Відстань: {e['distance']/1000:.2f} км<br>Вартість: {e['weight']:,.0f} грн")

        fig.add_trace(go.Scatter(x=mst_x, y=mst_y, mode='lines', line=dict(color=COLORS['edge_mst'], width=3.5), name='Лінії МКД'))
        fig.add_trace(go.Scatter(x=hover_x, y=hover_y, mode='markers', marker=dict(size=1, opacity=0), hovertext=hover_text, hoverinfo='text', showlegend=False))

    # резервні ребра (2-EC) — оранжевим пунктиром поверх МКД
    if aug_data and aug_data.get('reserve_edges'):
        rx, ry, rhx, rhy, rht = [], [], [], [], []
        for e in aug_data['reserve_edges']:
            p1, p2 = positions[e['node1']], positions[e['node2']]
            rx += [p1[0], p2[0], None]; ry += [p1[1], p2[1], None]
            rhx.append((p1[0] + p2[0]) / 2); rhy.append((p1[1] + p2[1]) / 2)
            rht.append(f"🛡️ РЕЗЕРВ<br>Відстань: {e['distance']/1000:.2f} км<br>Вартість: {e['weight']:,.0f} грн")
        fig.add_trace(go.Scatter(
            x=rx, y=ry, mode='lines',
            line=dict(color=COLORS['edge_reserve'], width=2.5, dash='dash'),
            name=f"Резервні лінії ({len(aug_data['reserve_edges'])})"
        ))
        fig.add_trace(go.Scatter(
            x=rhx, y=rhy, mode='markers',
            marker=dict(size=1, opacity=0),
            hovertext=rht, hoverinfo='text', showlegend=False
        ))

    nx, ny, nt = [p[0] for p in positions.values()], [p[1] for p in positions.values()], [n.name for n in g.iter_nodes()]
    fig.add_trace(go.Scatter(
        x=nx, y=ny, mode='markers+text' if g.node_count() <= 30 else 'markers', text=nt, textposition="top center",
        marker=dict(size=10, color=COLORS['node'], line=dict(width=1.5, color=COLORS['node_border'])), hoverinfo='text', hovertext=[f"<b>{n}</b><br>ID: {id}" for id, n in zip(positions.keys(), nt)]
    ))

    fig.update_layout(plot_bgcolor='white', margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor='x', scaleratio=1), legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(255,255,255,0.8)"))

    if mst_data:
        raw_name = mst_data['algorithm']
        if 'Indexed' in raw_name:
            algo_name = "Пріма (Indexed)"
        elif '-ary' in raw_name:
            # "Prim (4-ary Heap)" — d-арна купа. Дефіс відрізняє від "Binary".
            algo_name = "Пріма (d-ary)"
        elif 'Prim' in raw_name:
            algo_name = "Пріма (Binary)"
        else:
            algo_name = "Крускала"
        mem_kb = mst_data.get('peak_memory_kb')
        if mem_kb is None:
            mem_str = "-"
        elif mem_kb >= 1024:
            mem_str = f"{mem_kb/1024:.2f} МБ"
        else:
            mem_str = f"{mem_kb:,.1f} КБ"
        return fig, algo_name, f"{len(mst_data['edges'])}", f"{mst_data['total_distance']/1000:,.1f} км", f"{mst_data['total_cost']:,.0f} ₴", f"{mst_data['execution_time']*1000:.2f} мс", mem_str
    return fig, "-", "-", "-", "-", "-", "-"

# --- РЕДАКТОР КАРТИ ---
@app.callback(
    Output('store-editor-nodes', 'data', allow_duplicate=True), Output('store-editor-edges', 'data', allow_duplicate=True), Output('editor-map', 'center'),
    Input('btn-load-to-editor', 'n_clicks'), Input('btn-clear-editor', 'n_clicks'), State('store-main-graph', 'data'), prevent_initial_call=True
)
def sync_editor(load_clicks, clear_clicks, main_graph):
    ctx = callback_context
    if 'btn-clear-editor' in ctx.triggered[0]['prop_id']:
        return [], [], [48.46, 35.04]
    if not main_graph or not main_graph.get('nodes'): return [], [], no_update
    nodes = [{'id': n['id'], 'name': n['name'], 'x': n['x'], 'y': n['y']} for n in main_graph['nodes']]
    edges = [{'node1': e['node1'], 'node2': e['node2'], 'distance': e['distance'], 'cost': e.get('cost_per_meter', 150)} for e in main_graph['edges']]
    clat, clon = sum(n['x'] for n in nodes) / len(nodes) if nodes else 48.46, sum(n['y'] for n in nodes) / len(nodes) if nodes else 35.04
    return nodes, edges, [clat, clon]

@app.callback(
    Output('store-editor-nodes', 'data', allow_duplicate=True), Input('editor-map', 'clickData'), State('editor-mouse-mode', 'value'), State('store-editor-nodes', 'data'), prevent_initial_call=True
)
def map_click_add_node(click_data, mode, nodes):
    if mode != 'add_node' or not click_data: return no_update
    try: lat, lon = click_data['latlng']['lat'], click_data['latlng']['lng']
    except: return no_update
    new_nodes = list(nodes)
    new_id = max([n['id'] for n in new_nodes], default=-1) + 1
    new_nodes.append({'id': new_id, 'name': f"ПС-{new_id}", 'x': lat, 'y': lon})
    return new_nodes

@app.callback(
    Output('store-editor-nodes', 'data', allow_duplicate=True), Output('store-editor-edges', 'data', allow_duplicate=True), Output('store-selected-node', 'data', allow_duplicate=True), Output('editor-alerts', 'children'),
    Input({'type': 'editor-marker', 'index': ALL}, 'n_clicks'), State('editor-mouse-mode', 'value'), State('store-selected-node', 'data'), State('store-editor-nodes', 'data'), State('store-editor-edges', 'data'), prevent_initial_call=True
)
def marker_interaction(n_clicks_list, mode, selected_node, nodes, edges):
    ctx = dash.callback_context
    if not ctx.triggered or not any(n_clicks_list): return no_update, no_update, no_update, no_update
    try: clicked_id = json.loads(ctx.triggered[0]['prop_id'].split('.')[0])['index']
    except: return no_update, no_update, no_update, no_update

    new_nodes, new_edges = list(nodes), list(edges)

    if mode == 'delete_node':
        new_nodes = [n for n in new_nodes if n['id'] != clicked_id]
        new_edges = [e for e in new_edges if e['node1'] != clicked_id and e['node2'] != clicked_id]
        return new_nodes, new_edges, None, dbc.Alert("Підстанцію видалено!", color="warning", duration=2000)
    elif mode == 'add_edge':
        if selected_node is None: return no_update, no_update, clicked_id, dbc.Alert("Оберіть другу підстанцію!", color="info", duration=3000)
        if selected_node == clicked_id: return no_update, no_update, None, dbc.Alert("Скасовано.", color="secondary", duration=2000)
        k = (min(selected_node, clicked_id), max(selected_node, clicked_id))
        if any((min(e['node1'], e['node2']), max(e['node1'], e['node2'])) == k for e in new_edges): return no_update, no_update, None, dbc.Alert("Лінія вже існує!", color="danger", duration=2000)
        n1 = next((n for n in new_nodes if n['id'] == selected_node), None)
        n2 = next((n for n in new_nodes if n['id'] == clicked_id), None)
        if not n1 or not n2: return no_update, no_update, None, no_update
        lat1, lon1, lat2, lon2 = map(math.radians, [n1['x'], n1['y'], n2['x'], n2['y']])
        dist_m = 6371000 * 2 * math.asin(math.sqrt(math.sin((lat2-lat1)/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin((lon2-lon1)/2)**2))
        new_edges.append({'node1': selected_node, 'node2': clicked_id, 'distance': dist_m, 'cost': 150.0})
        return no_update, new_edges, None, dbc.Alert("Лінію створено!", color="success", duration=2000)
    return no_update, no_update, no_update, no_update

@app.callback(
    Output("modal-edit-edge", "is_open"), Output("modal-edge-dist", "value"), Output("modal-edge-cost", "value"), Output("store-editing-edge", "data"), Output("modal-edge-nodes-title", "children"),
    Input({'type': 'editor-line', 'index': ALL}, 'n_clicks'), State('editor-mouse-mode', 'value'), State('store-editor-edges', 'data'), State('store-editor-nodes', 'data'), prevent_initial_call=True
)
def open_edge_modal(n_clicks_list, mode, edges, nodes):
    ctx = dash.callback_context
    if not ctx.triggered or not any(n_clicks_list) or mode != 'pointer': return no_update, no_update, no_update, no_update, no_update
    try: n1_id, n2_id = map(int, json.loads(ctx.triggered[0]['prop_id'].split('.')[0])['index'].split('_'))
    except: return no_update, no_update, no_update, no_update, no_update
    edge = next((e for e in edges if (e['node1'] == n1_id and e['node2'] == n2_id) or (e['node1'] == n2_id and e['node2'] == n1_id)), None)
    if not edge: return no_update, no_update, no_update, no_update, no_update
    node_dict = {n['id']: n['name'] for n in nodes}
    return True, edge['distance'], edge['cost'], {'n1': n1_id, 'n2': n2_id}, f"{node_dict.get(n1_id, '?')} ↔ {node_dict.get(n2_id, '?')}"

@app.callback(
    Output('store-editor-edges', 'data', allow_duplicate=True), Output("modal-edit-edge", "is_open", allow_duplicate=True),
    Input("btn-modal-save-edge", "n_clicks"), Input("btn-modal-delete-edge", "n_clicks"), State("store-editing-edge", "data"), State("modal-edge-dist", "value"), State("modal-edge-cost", "value"), State("store-editor-edges", "data"), prevent_initial_call=True
)
def update_edge_from_modal(save_clicks, del_clicks, editing_edge, dist, cost, edges):
    ctx = dash.callback_context
    if not ctx.triggered or not editing_edge: return no_update, no_update
    trigger_id = ctx.triggered[0]['prop_id']
    new_edges, n1, n2 = list(edges), editing_edge['n1'], editing_edge['n2']
    if 'btn-modal-delete-edge' in trigger_id:
        return [e for e in new_edges if not ((e['node1'] == n1 and e['node2'] == n2) or (e['node1'] == n2 and e['node2'] == n1))], False
    elif 'btn-modal-save-edge' in trigger_id:
        for e in new_edges:
            if (e['node1'] == n1 and e['node2'] == n2) or (e['node1'] == n2 and e['node2'] == n1):
                e['distance'] = float(dist); e['cost'] = float(cost)
                break
        return new_edges, False
    return no_update, no_update

@app.callback(
    Output('editor-map-markers', 'children'), Output('editor-map-lines', 'children'), Output('editor-edges-table-container', 'children'),
    Input('store-editor-nodes', 'data'), Input('store-editor-edges', 'data'), Input('store-selected-node', 'data')
)
def update_editor_layers_and_table(nodes, edges, selected_node_id):
    markers = [dl.CircleMarker(id={'type': 'editor-marker', 'index': n['id']}, center=[n['x'], n['y']], radius=10 if n['id'] == selected_node_id else 8, color=COLORS['edge_new'] if n['id'] == selected_node_id else COLORS['node'], fillColor=COLORS['edge_new'] if n['id'] == selected_node_id else COLORS['node_new'], fillOpacity=0.9 if n['id'] == selected_node_id else 0.8, children=[dl.Tooltip(f"{n['name']} (ID: {n['id']})")]) for n in nodes]
    lines, table_rows, node_dict = [], [], {n['id']: n for n in nodes}
    for e in edges:
        if e['node1'] in node_dict and e['node2'] in node_dict:
            n1, n2 = node_dict[e['node1']], node_dict[e['node2']]
            lines.append(dl.Polyline(id={'type': 'editor-line', 'index': f"{n1['id']}_{n2['id']}"}, positions=[[n1['x'], n1['y']], [n2['x'], n2['y']]], color=COLORS['edge_mst'], weight=5, children=[dl.Tooltip(f"Відстань: {e['distance']/1000:.1f} км. Клікніть для ред.")]))
            table_rows.append(html.Tr([html.Td(n1['name']), html.Td(n2['name']), html.Td(f"{e['distance']/1000:.2f}"), html.Td(f"{e['cost']:.1f}"), html.Td(f"{e['distance']*e['cost']:,.0f}")]))
    table = dbc.Table([html.Thead(html.Tr([html.Th("ПС 1"), html.Th("ПС 2"), html.Th("Довж.(км)"), html.Th("Вар-ть"), html.Th("Сума")]))] + [html.Tbody(table_rows)], bordered=True, hover=True, size="sm") if table_rows else html.P("Лінії відсутні.", className="text-muted")
    return markers, lines, table

# --- АНІМАЦІЯ ---
@app.callback(Output('anim-interval', 'interval'), Input('dropdown-anim-speed', 'value'))
def update_anim_speed(speed_ms): return speed_ms

@app.callback(
    Output("anim-interval", "disabled"), Output("btn-anim-play", "children"), Output("btn-anim-play", "color"),
    Input("btn-anim-play", "n_clicks"), State("anim-interval", "disabled"), prevent_initial_call=True
)
def toggle_animation_play(n_clicks, disabled): return (False, "Пауза", "warning") if disabled else (True, "Відтворити", "success")

@app.callback(Output('slider-anim-step', 'max'), Output('slider-anim-step', 'marks'), Input('store-mst-result', 'data'))
def setup_anim_slider(mst_data):
    if not mst_data or not mst_data.get('steps'): return 1, {1: '1'}
    total = len(mst_data['steps'])
    return total, {1: '1', total: str(total), total//2: str(total//2)} if total > 10 else {1: '1', total: str(total)}

@app.callback(
    Output('slider-anim-step', 'value'), Output('anim-interval', 'disabled', allow_duplicate=True), Output('btn-anim-play', 'children', allow_duplicate=True), Output('btn-anim-play', 'color', allow_duplicate=True),
    Input('anim-interval', 'n_intervals'), State('slider-anim-step', 'value'), State('slider-anim-step', 'max'), prevent_initial_call=True
)
def advance_animation_frame(n, current_val, max_val):
    if current_val < max_val: return current_val + 1, no_update, no_update, no_update
    return no_update, True, "Відтворити", "success"

@app.callback(
    Output('slider-anim-step', 'value', allow_duplicate=True),
    Input('btn-anim-start', 'n_clicks'), Input('btn-anim-prev', 'n_clicks'), Input('btn-anim-next', 'n_clicks'), Input('btn-anim-end', 'n_clicks'),
    State('slider-anim-step', 'value'), State('slider-anim-step', 'max'), prevent_initial_call=True
)
def manual_anim_navigation(s, p, n, e, current_val, max_val):
    t = callback_context.triggered[0]['prop_id']
    if 'start' in t: return 1
    if 'prev' in t: return max(1, current_val - 1)
    if 'next' in t: return min(max_val, current_val + 1)
    if 'end' in t: return max_val
    return no_update

@app.callback(
    Output('plot-anim-graph', 'figure'), Output('anim-title', 'children'),
    Input('slider-anim-step', 'value'), State('store-main-graph', 'data'), State('store-mst-result', 'data')
)
def update_anim_frame(step_val, graph_data, mst_data):
    if not graph_data or not mst_data or not mst_data.get('steps'): return go.Figure(), "Побудуйте МКД"
    g, steps = dict_to_graph(graph_data), mst_data['steps']
    idx = min(step_val - 1, len(steps) - 1)
    cur_edges = steps[idx]
    new_edge = cur_edges[-1] if len(cur_edges) > len(steps[idx - 1] if idx > 0 else []) else None
    fig = go.Figure()
    pos = {n.id: (n.y, n.x) for n in g.iter_nodes()}
    bg_x, bg_y = [], []
    for e in g.iter_edges():
        bg_x += [pos[e.node1][0], pos[e.node2][0], None]; bg_y += [pos[e.node1][1], pos[e.node2][1], None]
    fig.add_trace(go.Scatter(x=bg_x, y=bg_y, mode='lines', line=dict(color=COLORS['edge_bg'], width=0.5), opacity=0.3, showlegend=False))
    mst_x, mst_y = [], []
    for e in cur_edges:
        if new_edge and e['node1'] == new_edge['node1'] and e['node2'] == new_edge['node2']: continue
        mst_x += [pos[e['node1']][0], pos[e['node2']][0], None]; mst_y += [pos[e['node1']][1], pos[e['node2']][1], None]
    if mst_x: fig.add_trace(go.Scatter(x=mst_x, y=mst_y, mode='lines', line=dict(color=COLORS['edge_mst'], width=3), name='Побудоване МКД'))
    if new_edge: fig.add_trace(go.Scatter(x=[pos[new_edge['node1']][0], pos[new_edge['node2']][0]], y=[pos[new_edge['node1']][1], pos[new_edge['node2']][1]], mode='lines', line=dict(color=COLORS['edge_new'], width=5), name='Поточний крок'))
    fig.add_trace(go.Scatter(x=[p[0] for p in pos.values()], y=[p[1] for p in pos.values()], mode='markers', marker=dict(size=8, color=COLORS['node']), showlegend=False))
    fig.update_layout(plot_bgcolor='white', margin=dict(l=10, r=10, t=10, b=10), xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor='x', scaleratio=1), legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    return fig, f"Алгоритм {'Пріма' if 'Prim' in mst_data.get('algorithm','') else 'Крускала'} | Крок {step_val} / {len(steps)} | Вартість: {sum(e['weight'] for e in cur_edges):,.0f} грн"

# --- БЕНЧМАРКИ ---
@app.callback(
    Output('bench-results-container', 'children'),
    Input('btn-run-bench', 'n_clicks'), State('dropdown-bench-type', 'value'), prevent_initial_call=True
)
def run_advanced_benchmarks(n_clicks, bench_type):
    bench = Benchmark()
    if bench_type == 'complexity':
        res = bench.full_comparison(sizes=[20, 50, 100, 200], verbose=False)
        p, k = res['PrimMST'], res['KruskalMST']
        f1 = go.Figure()
        f1.add_trace(go.Scatter(x=[20,50,100,200], y=[r.mean_time_ms for r in p], mode='lines+markers', name='Пріма', line=dict(color=COLORS['prim'], width=3)))
        f1.add_trace(go.Scatter(x=[20,50,100,200], y=[r.mean_time_ms for r in k], mode='lines+markers', name='Крускала', line=dict(color=COLORS['kruskal'], width=3)))
        f1.update_layout(title="Часова складність", xaxis_title="Кількість вершин (V)", yaxis_title="Час (мс)", plot_bgcolor='white', legend=dict(x=0.01, y=0.99))
        f2 = go.Figure()
        f2.add_trace(go.Scatter(x=[20,50,100,200], y=[r.mean_time_ms for r in p], mode='lines+markers', name='Пріма', line=dict(color=COLORS['prim'], width=3)))
        f2.add_trace(go.Scatter(x=[20,50,100,200], y=[r.mean_time_ms for r in k], mode='lines+markers', name='Крускала', line=dict(color=COLORS['kruskal'], width=3)))
        ty = [n**2 for n in [20,50,100,200]]
        factor = k[-1].mean_time_ms / ty[-1] if ty[-1] > 0 else 1
        f2.add_trace(go.Scatter(x=[20,50,100,200], y=[t * factor for t in ty], mode='lines', name='Теоретично O(V²)', line=dict(color='gray', width=2, dash='dash')))
        f2.update_layout(title="Log-Log шкала", xaxis_title="Вершин [Log]", yaxis_title="Час (мс) [Log]", plot_bgcolor='white', legend=dict(x=0.01, y=0.99))
        f2.update_xaxes(type="log"); f2.update_yaxes(type="log")
        return dbc.Row([dbc.Col(dcc.Graph(figure=f1), width=6), dbc.Col(dcc.Graph(figure=f2), width=6)])
    elif bench_type == 'density':
        res = bench.test_density_impact(node_count=150, verbose=False)
        p, k = res['PrimMST'], res['KruskalMST']
        dens = [r.density * 100 for r in p]
        f = go.Figure()
        f.add_trace(go.Scatter(x=dens, y=[r.mean_time_ms for r in p], mode='lines+markers', name='Пріма', line=dict(color=COLORS['prim'], width=3)))
        f.add_trace(go.Scatter(x=dens, y=[r.mean_time_ms for r in k], mode='lines+markers', name='Крускала', line=dict(color=COLORS['kruskal'], width=3)))
        f.update_layout(title="Вплив щільності графа на швидкість (Crossover Point)", xaxis_title="Щільність ребер (%)", yaxis_title="Час (мс)", plot_bgcolor='white')
        return dbc.Row([dbc.Col(dcc.Graph(figure=f), width=12)])
    elif bench_type == 'dynamic':
        res = bench.test_dynamic_vs_static(node_count=300, verbose=False)
        f = go.Figure()
        f.add_trace(go.Bar(x=['Повна перебудова', 'Інкрементально'], y=[res['static_ms'], res['dynamic_ms']], marker_color=[COLORS['prim'], COLORS['edge_mst']], text=[f"{res['static_ms']:.2f} мс", f"{res['dynamic_ms']:.2f} мс"], textposition='auto'))
        f.update_layout(title="Ефективність динамічного оновлення (Лог. шкала)", yaxis_title="Час (мс)", plot_bgcolor='white'); f.update_yaxes(type="log")
        s = res['static_ms'] / res['dynamic_ms'] if res['dynamic_ms'] > 0 else 0
        return html.Div([dbc.Row([dbc.Col(dcc.Graph(figure=f), width=12)]), dbc.Alert(f"Динамічне оновлення швидше у {s:,.0f} разів", color="success", className="mt-3 text-center fw-bold")])

    elif bench_type == 'reference':
        # Референс-тест рецензента: розріджений граф V=500, E=4000
        from algorithms import PrimMSTDAry, PrimMSTIndexed
        gen = GraphGenerator(seed=42)
        sparse_graph = gen.sparse_graph(num_nodes=500, num_edges=4000)
        density = 2 * sparse_graph.edge_count() / (sparse_graph.node_count() * (sparse_graph.node_count() - 1))

        variants = [
            ('Прім (Binary)', PrimMST, COLORS['prim']),
            ('Прім (4-ary)', lambda g, **kw: PrimMSTDAry(g, d=4, **kw), '#9467BD'),
            ('Прім (Indexed)', PrimMSTIndexed, '#17BECF'),
            ('Крускал', KruskalMST, COLORS['kruskal']),
        ]
        names, times, mems, costs = [], [], [], []
        for name, klass, _ in variants:
            if name == 'Прім (4-ary)':
                r = bench._measure_with_factory(klass, sparse_graph, name,
                                                 verbose=False, memory_iterations=5)
            else:
                r = bench.measure_full(klass, sparse_graph, verbose=False,
                                       memory_iterations=5)
            names.append(name)
            times.append(r.mean_time_ms)
            mems.append(r.peak_memory_kb)
            costs.append(r.mst_cost)

        f1 = go.Figure()
        f1.add_trace(go.Bar(
            x=names, y=times,
            marker_color=[v[2] for v in variants],
            text=[f"{t:.2f} мс" for t in times], textposition='auto'
        ))
        f1.update_layout(
            title=f"Розріджений граф: V={sparse_graph.node_count()}, "
                  f"E={sparse_graph.edge_count()} (щільність {density:.1%})",
            yaxis_title="Час (мс)", plot_bgcolor='white', showlegend=False
        )

        f2 = go.Figure()
        f2.add_trace(go.Bar(
            x=names, y=mems,
            marker_color=[v[2] for v in variants],
            text=[f"{m:,.0f} КБ" for m in mems], textposition='auto'
        ))
        f2.update_layout(title="Пам'ять (пік)", yaxis_title="КБ",
                          plot_bgcolor='white', showlegend=False)

        all_match = max(costs) - min(costs) < 0.01
        return html.Div([
            dbc.Row([
                dbc.Col(dcc.Graph(figure=f1), width=7),
                dbc.Col(dcc.Graph(figure=f2), width=5),
            ]),
            dbc.Alert(
                f"Вартість МКД у всіх алгоритмів збігається: {costs[0]:,.0f} грн"
                if all_match else
                f"Розбіжність вартостей: {min(costs):,.0f}..{max(costs):,.0f} грн",
                color="success" if all_match else "warning",
                className="mt-2 text-center"
            ),
        ])

    elif bench_type == 'complexity_full':
        # Повна масштабованість до 1000 ПС (повільно)
        sizes = [50, 100, 200, 500, 1000]
        res = bench.full_comparison(sizes=sizes, verbose=False)
        p, k = res['PrimMST'], res['KruskalMST']

        f1 = go.Figure()
        f1.add_trace(go.Scatter(x=sizes, y=[r.mean_time_ms for r in p],
                                mode='lines+markers', name='Прім',
                                line=dict(color=COLORS['prim'], width=3)))
        f1.add_trace(go.Scatter(x=sizes, y=[r.mean_time_ms for r in k],
                                mode='lines+markers', name='Крускал',
                                line=dict(color=COLORS['kruskal'], width=3)))
        f1.update_layout(title="Час vs кількість вершин (V=50..1000)",
                          xaxis_title="V", yaxis_title="Час (мс)",
                          plot_bgcolor='white')

        f2 = go.Figure()
        f2.add_trace(go.Scatter(x=sizes, y=[r.peak_memory_kb for r in p],
                                mode='lines+markers', name='Прім',
                                line=dict(color=COLORS['prim'], width=3)))
        f2.add_trace(go.Scatter(x=sizes, y=[r.peak_memory_kb for r in k],
                                mode='lines+markers', name='Крускал',
                                line=dict(color=COLORS['kruskal'], width=3)))
        f2.update_layout(title="Пам'ять vs кількість вершин",
                          xaxis_title="V", yaxis_title="Пам'ять (КБ)",
                          plot_bgcolor='white')

        return html.Div([
            dbc.Row([
                dbc.Col(dcc.Graph(figure=f1), width=6),
                dbc.Col(dcc.Graph(figure=f2), width=6),
            ]),
        ])

    elif bench_type == 'prim_variants':
        # Порівняння варіантів куп Пріма на різних розмірах
        sizes = [50, 100, 200, 500]
        variants_res = bench.compare_prim_variants(
            sizes=sizes, graph_type='random', dary_d=4, verbose=False
        )
        color_map = {
            'Prim (Binary)': COLORS['prim'],
            'Prim (4-ary)': '#9467BD',
            'Prim (Indexed)': '#17BECF',
            'Kruskal': COLORS['kruskal'],
        }
        f = go.Figure()
        for name, lst in variants_res.items():
            f.add_trace(go.Scatter(
                x=[r.node_count for r in lst],
                y=[r.mean_time_ms for r in lst],
                mode='lines+markers', name=name,
                line=dict(color=color_map.get(name, '#888'), width=3),
            ))
        f.update_layout(
            title="Порівняння варіантів реалізації Пріма (повний граф)",
            xaxis_title="Кількість вершин (V)", yaxis_title="Час (мс)",
            plot_bgcolor='white',
        )

        fmem = go.Figure()
        for name, lst in variants_res.items():
            fmem.add_trace(go.Scatter(
                x=[r.node_count for r in lst],
                y=[r.peak_memory_kb for r in lst],
                mode='lines+markers', name=name,
                line=dict(color=color_map.get(name, '#888'), width=3),
            ))
        fmem.update_layout(
            title="Пам'ять (пік) — Indexed PQ значно економніший",
            xaxis_title="V", yaxis_title="Пам'ять (КБ)",
            plot_bgcolor='white',
        )

        return html.Div([
            dbc.Row([
                dbc.Col(dcc.Graph(figure=f), width=6),
                dbc.Col(dcc.Graph(figure=fmem), width=6),
            ]),
        ])

# --- ЕКСПОРТ ---
@app.callback(Output("download-json", "data"), Input("btn-export-json", "n_clicks"), State("store-mst-result", "data"), prevent_initial_call=True)
def export_json(n_clicks, mst_data): return dcc.send_string(json.dumps(mst_data, indent=2, ensure_ascii=False), "mst_result.json") if mst_data else no_update

@app.callback(Output("download-csv", "data"), Input("btn-export-csv", "n_clicks"), State("store-mst-result", "data"), prevent_initial_call=True)
def export_csv(n_clicks, mst_data):
    if not mst_data: return no_update
    import pandas as pd
    return dcc.send_data_frame(pd.DataFrame(mst_data['edges']).to_csv, "mst_edges.csv", index=False)

@app.callback(Output("download-geojson", "data"), Input("btn-export-geojson", "n_clicks"), State("store-main-graph", "data"), State("store-mst-result", "data"), prevent_initial_call=True)
def export_geojson(n_clicks, g_data, mst_data):
    if not g_data or not mst_data: return no_update
    ft = [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [n['y'], n['x']]}, "properties": {"name": n['name']}} for n in g_data['nodes']]
    nd = {n['id']: n for n in g_data['nodes']}
    ft += [{"type": "Feature", "geometry": {"type": "LineString", "coordinates": [[nd[e['node1']]['y'], nd[e['node1']]['x']], [nd[e['node2']]['y'], nd[e['node2']]['x']]]}} for e in mst_data['edges']]
    return dcc.send_string(json.dumps({"type": "FeatureCollection", "features": ft}, ensure_ascii=False), "mst_network.geojson")

# --- РОЗШИРЕНИЙ РЕДАКТОР ---
@app.callback(Output('table-adv-nodes', 'data'), Output('table-adv-edges', 'data'), Input('btn-adv-load', 'n_clicks'), State('store-main-graph', 'data'), prevent_initial_call=True)
def load_adv_tables(n_clicks, main_graph):
    if not main_graph: return [], []
    return [{'id': n['id'], 'name': n['name'], 'x': n['x'], 'y': n['y']} for n in main_graph['nodes']], [{'node1': e['node1'], 'node2': e['node2'], 'distance': e['distance'], 'cost_per_meter': e.get('cost_per_meter', 150)} for e in main_graph['edges']]

@app.callback(Output('table-adv-nodes', 'data', allow_duplicate=True), Input('btn-adv-add-node', 'n_clicks'), State('table-adv-nodes', 'data'), prevent_initial_call=True)
def add_adv_node_row(n_clicks, rows):
    rows = rows or []; rows.append({'id': max([int(r.get('id', -1)) for r in rows], default=-1) + 1, 'name': f"Нова ПС", 'x': 48.5, 'y': 35.0})
    return rows

@app.callback(Output('table-adv-edges', 'data', allow_duplicate=True), Input('btn-adv-add-edge', 'n_clicks'), State('table-adv-edges', 'data'), prevent_initial_call=True)
def add_adv_edge_row(n_clicks, rows):
    rows = rows or []; rows.append({'node1': 0, 'node2': 1, 'distance': 1000.0, 'cost_per_meter': 150.0}); return rows

@app.callback(Output('table-adv-edges', 'data', allow_duplicate=True), Output('adv-alerts', 'children', allow_duplicate=True), Input('btn-adv-recalc', 'n_clicks'), State('table-adv-nodes', 'data'), State('table-adv-edges', 'data'), State('dropdown-adv-method', 'value'), prevent_initial_call=True)
def recalc_adv_distances(n_clicks, nodes, edges, method):
    if not nodes or not edges: return no_update, dbc.Alert("Немає даних.", color="warning")
    node_dict = {int(n['id']): n for n in nodes if n.get('id') is not None}
    for e in edges:
        try:
            n1, n2 = int(e['node1']), int(e['node2'])
            if n1 in node_dict and n2 in node_dict:
                lat1, lon1, lat2, lon2 = float(node_dict[n1]['x']), float(node_dict[n1]['y']), float(node_dict[n2]['x']), float(node_dict[n2]['y'])
                if method == 'haversine':
                    l1, ln1, l2, ln2 = map(math.radians, [lat1, lon1, lat2, lon2])
                    e['distance'] = round(6371000 * 2 * math.asin(math.sqrt(math.sin((l2 - l1) / 2) ** 2 + math.cos(l1) * math.cos(l2) * math.sin((ln2 - ln1) / 2) ** 2)), 2)
                else:
                    e['distance'] = round(math.sqrt(((lat2 - lat1) * 111320) ** 2 + ((lon2 - lon1) * 111320 * math.cos(math.radians((lat1 + lat2) / 2))) ** 2), 2)
        except: pass
    return edges, dbc.Alert("Відстані перераховано!", color="success", duration=3000)

@app.callback(Output('store-main-graph', 'data', allow_duplicate=True), Output('adv-alerts', 'children'), Input('btn-adv-save', 'n_clicks'), State('table-adv-nodes', 'data'), State('table-adv-edges', 'data'), prevent_initial_call=True)
def save_adv_to_main(n_clicks, node_rows, edge_rows):
    try:
        g = Graph()
        for r in node_rows:
            if r.get('id') is not None: g.add_node(Node(int(r['id']), str(r.get('name', '')), float(r['x']), float(r['y'])))
        for r in edge_rows:
            if r.get('node1') is not None and r.get('node2') is not None: g.add_edge(Edge(int(r['node1']), int(r['node2']), float(r.get('distance', 0)), float(r.get('cost_per_meter', 150))))
        return graph_to_dict(g), dbc.Alert("Граф збережено!", color="success", duration=3000)
    except Exception as e: return no_update, dbc.Alert(f"Помилка: {e}", color="danger")

if __name__ == '__main__':
    app.run(debug=True)