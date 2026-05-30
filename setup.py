from data.database import DatabaseManager
from data.generator import GraphGenerator
from data.demo import create_demo_graph, COST_PER_METER
from algorithms import PrimMST, KruskalMST

db = DatabaseManager("energy_network.db")

# 1. Очистити все
for net in db.list_networks():
    db.delete_network(net['id'])
db.clear_benchmarks()
print("База очищена")


# 3. Кластерна мережа (3 міста по 8 підстанцій)
gen = GraphGenerator(seed=42)
cluster = gen.cluster_graph(num_clusters=3, nodes_per_cluster=8, cost_per_meter=150.0)
net_id = db.save_network("Кластерна мережа (3x8)", cluster, cost_per_meter=150.0)
mst = PrimMST(cluster).find_mst()
db.save_mst_result(net_id, mst, algorithm=mst.algorithm_name)
print(f"+ Кластерна: {cluster.node_count()} вершин, МКД = {mst.total_cost:,.0f} грн")

# 4. Сіткова мережа 5×5
grid = gen.grid_graph(rows=5, cols=5, cost_per_meter=150.0)
net_id = db.save_network("Сітка 5×5", grid, cost_per_meter=150.0)
mst = PrimMST(grid).find_mst()
db.save_mst_result(net_id, mst, algorithm=mst.algorithm_name)
print(f"+ Сiтка 5x5: {grid.node_count()} вершин, МКД = {mst.total_cost:,.0f} грн")

# 5. Великий випадковий граф
big = gen.random_graph(num_nodes=100, cost_per_meter=150.0, complete=True)
net_id = db.save_network("Випадковий (100 ПС)", big, cost_per_meter=150.0)
prim = PrimMST(big).find_mst(track_memory=True)
kruskal = KruskalMST(big).find_mst(track_memory=True)
db.save_mst_result(net_id, prim, algorithm=prim.algorithm_name)
db.save_mst_result(net_id, kruskal, algorithm=kruskal.algorithm_name)
print(f"+ Випадковий: {big.node_count()} вершин, МКД = {prim.total_cost:,.0f} грн, "
      f"пам'ять Прім={prim.peak_memory_kb:.1f} КБ")

# 6. Розріджена 500-ПС мережа (референс рецензента)
sparse500 = gen.sparse_graph(num_nodes=500, num_edges=4000, cost_per_meter=150.0)
net_id = db.save_network("Розріджена (500 ПС, 4000 ребер)", sparse500, cost_per_meter=150.0)
prim = PrimMST(sparse500).find_mst(track_memory=True)
kruskal = KruskalMST(sparse500).find_mst(track_memory=True)
db.save_mst_result(net_id, prim, algorithm=prim.algorithm_name)
db.save_mst_result(net_id, kruskal, algorithm=kruskal.algorithm_name)
print(f"+ Розріджена 500: МКД = {prim.total_cost:,.0f} грн, "
      f"Прім {prim.execution_time*1000:.1f}мс / Крускал {kruskal.execution_time*1000:.1f}мс")

# 7. Великий повний граф 1000 ПС
huge = gen.random_graph(num_nodes=1000, cost_per_meter=150.0, complete=True)
net_id = db.save_network("Повний (1000 ПС)", huge, cost_per_meter=150.0)
prim = PrimMST(huge).find_mst(track_memory=True)
kruskal = KruskalMST(huge).find_mst(track_memory=True)
db.save_mst_result(net_id, prim, algorithm=prim.algorithm_name)
db.save_mst_result(net_id, kruskal, algorithm=kruskal.algorithm_name)
print(f"+ Повний 1000: {huge.edge_count()} ребер, "
      f"Прім {prim.execution_time*1000:.1f}мс / Крускал {kruskal.execution_time*1000:.1f}мс, "
      f"пам'ять {prim.peak_memory_mb:.1f} МБ")

# Перевірка
print(f"\nУ базі: {len(db.list_networks())} мереж")
for net in db.list_networks():
    print(f"  [{net['id']}] {net['name']}")